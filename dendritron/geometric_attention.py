"""Causal Euclidean HarMax derivative contraction without Q/K/V or softmax."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.geometric_attention requires PyTorch. Install torch>=2.7."
    ) from error


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, values: Tensor) -> Tensor:
        energy = values.square().mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(energy + self.epsilon) * self.weight


@dataclass(frozen=True)
class HarMaxContractionStats:
    selected_positions: Tensor
    squared_distances: Tensor
    distance_mass: Tensor
    target_mass: Tensor
    signed_coefficients: Tensor
    attraction_mass: Tensor
    repulsion_mass: Tensor
    harmonic_residual: Tensor
    confidence: Tensor


class HarMaxContraction(nn.Module):
    """Sparse causal movement from the negative HarMax-loss derivative.

    Pool mass comes from ordinary Euclidean inverse distances. Target mass
    comes from positive evidence strengths. Their difference
    ``target_mass - distance_mass`` creates attraction and repulsion directly.
    """

    def __init__(
        self,
        width: int,
        *,
        max_sequence_length: int,
        candidate_window: int,
        top_k: int,
        harmonic_exponent: float = 2.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if (
            width < 1
            or max_sequence_length < 1
            or candidate_window < 1
            or top_k < 1
        ):
            raise ValueError(
                "width, max_sequence_length, candidate_window, and top_k "
                "must be positive"
            )
        if candidate_window > max_sequence_length:
            raise ValueError("candidate_window cannot exceed max_sequence_length")
        if top_k > candidate_window:
            raise ValueError("top_k cannot exceed candidate_window")
        if harmonic_exponent <= 0:
            raise ValueError("harmonic_exponent must be positive")
        self.width = int(width)
        self.max_sequence_length = int(max_sequence_length)
        self.candidate_window = int(candidate_window)
        self.top_k = int(top_k)
        self.harmonic_exponent = float(harmonic_exponent)
        self.epsilon = float(epsilon)

        self.input_norm = RMSNorm(width, epsilon)
        self.relative_evidence = nn.Parameter(torch.zeros(candidate_window))
        self.relative_penalty_raw = nn.Parameter(torch.tensor(0.0))

    def _evaluate_pool(
        self,
        query: Tensor,
        anchors: Tensor,
        evidence: Tensor,
        supported: Tensor,
        valid: Tensor,
        relative_displacement: Tensor,
    ) -> tuple[Tensor, ...]:
        displacement = anchors - query.unsqueeze(-2)
        position_penalty = F.softplus(self.relative_penalty_raw)
        squared_distances = (
            displacement.square().sum(dim=-1)
            + position_penalty * relative_displacement.square()
            + self.epsilon**2
        )

        inverse_distance = squared_distances.pow(
            -0.5 * self.harmonic_exponent
        )
        inverse_distance = inverse_distance * valid
        distance_mass = inverse_distance / (
            inverse_distance.sum(dim=-1, keepdim=True) + self.epsilon
        )

        positive_evidence = evidence.clamp_min(0.0) * supported * valid
        if bool((positive_evidence.sum(dim=-1) <= 0).any()):
            raise ValueError(
                "Every executable HarMax pool requires positive supported evidence"
            )
        target_mass = positive_evidence / (
            positive_evidence.sum(dim=-1, keepdim=True) + self.epsilon
        )
        signed_coefficients = target_mass - distance_mass
        movement = self.harmonic_exponent * (
            signed_coefficients.unsqueeze(-1)
            * displacement
            / squared_distances.unsqueeze(-1)
        ).sum(dim=-2)
        harmonic_residual = -(
            target_mass * distance_mass.clamp_min(self.epsilon).log()
        ).sum(dim=-1)
        confidence = (1.0 + harmonic_residual).reciprocal()
        attraction_mass = signed_coefficients.clamp_min(0.0).sum(dim=-1)
        repulsion_mass = (-signed_coefficients.clamp_max(0.0)).sum(dim=-1)
        return (
            movement,
            squared_distances,
            distance_mass,
            target_mass,
            signed_coefficients,
            attraction_mass,
            repulsion_mass,
            harmonic_residual,
            confidence,
        )

    def contract_pool(
        self,
        query: Tensor,
        anchors: Tensor,
        *,
        evidence: Tensor,
        supported: Tensor,
        valid: Tensor | None = None,
        relative_displacement: Tensor | None = None,
    ) -> tuple[Tensor, HarMaxContractionStats]:
        """Evaluate an explicitly bound memory or reasoning pool."""

        if query.ndim != 3 or query.shape[-1] != self.width:
            raise ValueError(f"query must be [B,T,{self.width}]")
        if anchors.ndim != 4 or anchors.shape[:2] != query.shape[:2]:
            raise ValueError("anchors must be [B,T,Q,D]")
        if anchors.shape[-1] != self.width:
            raise ValueError("anchor width differs from the configured width")
        pool_shape = anchors.shape[:-1]
        if evidence.shape != pool_shape or supported.shape != pool_shape:
            raise ValueError("evidence and supported must be [B,T,Q]")
        if valid is None:
            valid = torch.ones(pool_shape, dtype=torch.bool, device=query.device)
        if valid.shape != pool_shape:
            raise ValueError("valid must be [B,T,Q]")
        if relative_displacement is None:
            relative_displacement = query.new_zeros(pool_shape)
        if relative_displacement.shape != pool_shape:
            raise ValueError("relative_displacement must be [B,T,Q]")

        normalized_query = self.input_norm(query)
        normalized_anchors = self.input_norm(anchors)
        result = self._evaluate_pool(
            normalized_query,
            normalized_anchors,
            evidence.to(normalized_query),
            supported.to(device=query.device, dtype=torch.bool),
            valid.to(device=query.device, dtype=torch.bool),
            relative_displacement.to(normalized_query),
        )
        movement = result[0]
        pool_size = anchors.shape[-2]
        positions = torch.arange(
            pool_size,
            device=query.device,
            dtype=torch.long,
        ).view(1, 1, pool_size).expand(*pool_shape)
        positions = positions.masked_fill(~valid, -1)
        return movement, HarMaxContractionStats(
            selected_positions=positions,
            squared_distances=result[1],
            distance_mass=result[2],
            target_mass=result[3],
            signed_coefficients=result[4],
            attraction_mass=result[5],
            repulsion_mass=result[6],
            harmonic_residual=result[7],
            confidence=result[8],
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, HarMaxContractionStats]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError(
                f"Expected hidden [B,T,{self.width}], found {tuple(hidden.shape)}"
            )
        length = hidden.shape[1]
        if length < 1:
            raise ValueError("HarMax contraction requires at least one position")
        if length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {length} exceeds {self.max_sequence_length}"
            )

        normalized = self.input_norm(hidden)
        window = min(self.candidate_window, max(length - 1, 1))
        positions = torch.arange(length, device=hidden.device)
        offsets = torch.arange(1, window + 1, device=hidden.device)
        candidate_positions = positions[:, None] - offsets[None, :]
        candidate_valid = candidate_positions >= 0
        safe_positions = candidate_positions.clamp(min=0)

        # The first position has no causal predecessor.  Its self anchor forms
        # a one-item balanced pool and therefore yields exactly zero movement.
        candidate_valid[0, 0] = True
        safe_positions[0, 0] = 0

        candidates = normalized[:, safe_positions, :]
        query = normalized
        displacement = candidates - query.unsqueeze(-2)
        position_penalty = F.softplus(self.relative_penalty_raw)
        relative = offsets.to(hidden.dtype) / max(window, 1)
        squared_distances = (
            displacement.square().sum(dim=-1)
            + position_penalty * relative.square().view(1, 1, window)
            + self.epsilon**2
        )
        valid = candidate_valid.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        selected_count = min(self.top_k, window)
        selected_local = squared_distances.masked_fill(
            ~valid,
            torch.inf,
        ).topk(selected_count, largest=False, dim=-1).indices
        selected_valid = valid.gather(-1, selected_local)
        selected_anchors = candidates.gather(
            -2,
            selected_local.unsqueeze(-1).expand(
                *selected_local.shape,
                self.width,
            ),
        )
        selected_positions = safe_positions.unsqueeze(0).expand(
            hidden.shape[0], -1, -1
        ).gather(-1, selected_local)
        selected_positions = selected_positions.masked_fill(~selected_valid, -1)

        evidence_by_offset = F.softplus(self.relative_evidence[:window])
        evidence = evidence_by_offset.view(1, 1, window).expand_as(
            squared_distances
        ).gather(-1, selected_local)
        relative_selected = relative.view(1, 1, window).expand_as(
            squared_distances
        ).gather(-1, selected_local)
        supported = selected_valid
        result = self._evaluate_pool(
            query,
            selected_anchors,
            evidence,
            supported,
            selected_valid,
            relative_selected,
        )
        update = result[0]

        if return_stats:
            return update, HarMaxContractionStats(
                selected_positions=selected_positions,
                squared_distances=result[1],
                distance_mass=result[2],
                target_mass=result[3],
                signed_coefficients=result[4],
                attraction_mass=result[5],
                repulsion_mass=result[6],
                harmonic_residual=result[7],
                confidence=result[8],
            )
        return update
