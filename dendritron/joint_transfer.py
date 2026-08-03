"""Shared latent reference frame for Dendritron memory and live states.

The canonical joint domain is the frozen Qwen layer-2 definition geometry.
Definition tensors pass through unchanged.  Surface words, token IDs, and
sense IDs remain lookup metadata outside this module.  Trainable transfer
maps place layer-8 Engrams, layer-24 Engrams, and the live recurrent state in
the same coordinate frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.joint_transfer requires PyTorch. Install torch>=2.7."
    ) from error


TransferSource = Literal["layer8", "layer24", "live"]


def _identity_or_orthogonal(weight: Tensor) -> None:
    """Initialize a linear map as identity whenever its widths agree."""

    if weight.shape[0] == weight.shape[1]:
        nn.init.eye_(weight)
    else:
        nn.init.orthogonal_(weight)


@dataclass(frozen=True)
class JointTransferAlignmentStats:
    point_loss: Tensor
    locality_loss: Tensor
    total_loss: Tensor


class JointTransferDomain(nn.Module):
    """Map every non-reference source into layer-2 definition geometry.

    The definition bank itself defines the reference frame, so
    :meth:`definitions_to_joint` is an exact identity operation.  The three
    learned source maps are fitted from same-content anchor pairs.  A final
    joint-to-live map lets a joint-space movement update a recipient whose
    internal coordinates have not yet converged to the reference frame.
    """

    def __init__(
        self,
        live_width: int,
        *,
        memory_width: int,
        joint_width: int | None = None,
    ) -> None:
        super().__init__()
        self.live_width = int(live_width)
        self.memory_width = int(memory_width)
        self.joint_width = (
            self.memory_width if joint_width is None else int(joint_width)
        )
        if min(self.live_width, self.memory_width, self.joint_width) < 1:
            raise ValueError("JTD widths must be positive")
        if self.joint_width != self.memory_width:
            raise ValueError(
                "The joint width must equal the layer-2 definition width so "
                "definition locations remain unchanged"
            )

        self.layer8_to_joint = nn.Linear(
            self.memory_width,
            self.joint_width,
            bias=False,
        )
        self.layer24_to_joint = nn.Linear(
            self.memory_width,
            self.joint_width,
            bias=False,
        )
        self.live_to_joint = nn.Linear(
            self.live_width,
            self.joint_width,
            bias=False,
        )
        self.joint_to_live = nn.Linear(
            self.joint_width,
            self.live_width,
            bias=False,
        )
        for mapping in (
            self.layer8_to_joint,
            self.layer24_to_joint,
            self.live_to_joint,
            self.joint_to_live,
        ):
            _identity_or_orthogonal(mapping.weight)

    def definitions_to_joint(self, values: Tensor) -> Tensor:
        """Return frozen definition vectors at their original locations."""

        if values.shape[-1] != self.memory_width:
            raise ValueError(
                f"Definition width must be {self.memory_width}, found "
                f"{values.shape[-1]}"
            )
        return values

    def source_to_joint(self, values: Tensor, source: TransferSource) -> Tensor:
        expected = self.live_width if source == "live" else self.memory_width
        if values.shape[-1] != expected:
            raise ValueError(
                f"{source} width must be {expected}, found {values.shape[-1]}"
            )
        if source == "layer8":
            return self.layer8_to_joint(values)
        if source == "layer24":
            return self.layer24_to_joint(values)
        if source == "live":
            return self.live_to_joint(values)
        raise ValueError(f"Unknown JTD source: {source}")

    def movement_to_live(self, values: Tensor) -> Tensor:
        if values.shape[-1] != self.joint_width:
            raise ValueError(
                f"Joint movement width must be {self.joint_width}, found "
                f"{values.shape[-1]}"
            )
        return self.joint_to_live(values)

    @torch.no_grad()
    def load_projection(self, source: TransferSource, weight: Tensor) -> None:
        mapping = {
            "layer8": self.layer8_to_joint,
            "layer24": self.layer24_to_joint,
            "live": self.live_to_joint,
        }[source]
        if tuple(weight.shape) != tuple(mapping.weight.shape):
            raise ValueError(
                f"{source} projection must have shape "
                f"{tuple(mapping.weight.shape)}"
            )
        mapping.weight.copy_(
            weight.to(device=mapping.weight.device, dtype=mapping.weight.dtype)
        )


def locality_preserving_alignment_loss(
    projected: Tensor,
    reference: Tensor,
    *,
    neighbor_count: int = 8,
    locality_weight: float = 1.0,
    epsilon: float = 1e-6,
) -> tuple[Tensor, JointTransferAlignmentStats]:
    """Fit same-content states while preserving reference neighborhoods.

    ``reference`` contains layer-2 anchor locations.  Rows in ``projected``
    correspond to the same texts from another source.  The point term aligns
    each pair and the locality term preserves nearby layer-2 relationships.
    Both terms use Euclidean distances in the fixed reference frame.
    """

    if projected.ndim != 2 or reference.ndim != 2:
        raise ValueError("Alignment tensors must be [N,D]")
    if projected.shape != reference.shape:
        raise ValueError("Projected and reference anchor shapes must match")
    if projected.shape[0] < 2:
        raise ValueError("Alignment requires at least two anchor pairs")
    if neighbor_count < 1:
        raise ValueError("neighbor_count must be positive")

    point_loss = F.mse_loss(projected, reference)
    count = min(int(neighbor_count), reference.shape[0] - 1)
    with torch.no_grad():
        reference_distances = torch.cdist(reference, reference, p=2)
        diagonal = torch.eye(
            reference.shape[0],
            dtype=torch.bool,
            device=reference.device,
        )
        neighbor_indices = reference_distances.masked_fill(
            diagonal,
            torch.inf,
        ).topk(count, largest=False, dim=-1).indices
        target = reference_distances.gather(-1, neighbor_indices)
        target_scale = target.mean().clamp_min(float(epsilon))

    projected_distances = torch.cdist(projected, projected, p=2)
    observed = projected_distances.gather(-1, neighbor_indices)
    locality_loss = F.smooth_l1_loss(
        observed / target_scale,
        target / target_scale,
    )
    total = point_loss + float(locality_weight) * locality_loss
    return total, JointTransferAlignmentStats(
        point_loss=point_loss,
        locality_loss=locality_loss,
        total_loss=total,
    )
