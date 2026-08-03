"""Sparse memory fusion in the layer-2 joint concept geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.memory_fusion requires PyTorch. Install torch>=2.7."
    ) from error

from .joint_transfer import JointTransferDomain


@dataclass
class MemoryPayloads:
    """Rows materialized by the CPU surface-address layer.

    Phrase tensors contain zero rows at exact-memory misses.  Definition rows
    retain a sense axis and already occupy the canonical Qwen layer-2 concept
    space.  ``definition_sense_rows`` contains trace metadata only; it never
    enters a latent vector.  Hash addresses use ``-1`` at unavailable
    positions.
    """

    phrase_layer8: Tensor | None = None
    phrase_layer24: Tensor | None = None
    phrase_mask: Tensor | None = None
    definitions: Tensor | None = None
    definition_mask: Tensor | None = None
    definition_sense_rows: Tensor | None = None
    hash_addresses: Mapping[int, Tensor] | None = None

    def to(self, device: torch.device | str) -> "MemoryPayloads":
        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device)

        return MemoryPayloads(
            phrase_layer8=move(self.phrase_layer8),
            phrase_layer24=move(self.phrase_layer24),
            phrase_mask=move(self.phrase_mask),
            definitions=move(self.definitions),
            definition_mask=move(self.definition_mask),
            definition_sense_rows=move(self.definition_sense_rows),
            hash_addresses=(
                None
                if self.hash_addresses is None
                else {
                    order: value.to(device)
                    for order, value in self.hash_addresses.items()
                }
            ),
        )


@dataclass(frozen=True)
class MemoryFusionStats:
    phrase8_gate: Tensor
    phrase24_gate: Tensor
    definition_gate: Tensor
    hash_gate: Tensor
    definition_weights: Tensor | None
    definition_squared_distances: Tensor | None
    definition_sense_rows: Tensor | None
    active_definition_count: Tensor | None


class TrainableHashEngram(nn.Module):
    def __init__(
        self,
        model_width: int,
        *,
        rows_by_order: Mapping[int, int],
        heads: int,
        memory_width: int,
        initialization_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.model_width = int(model_width)
        self.heads = int(heads)
        self.memory_width = int(memory_width)
        self.initialization_gain = float(initialization_gain)
        if self.initialization_gain <= 0:
            raise ValueError("initialization_gain must be positive")
        self.rows_by_order = {int(k): int(v) for k, v in rows_by_order.items()}
        self.tables = nn.ModuleDict(
            {
                str(order): nn.Embedding(rows, memory_width)
                for order, rows in self.rows_by_order.items()
            }
        )
        self.projections = nn.ModuleDict(
            {
                str(order): nn.Linear(memory_width, model_width, bias=False)
                for order in self.rows_by_order
            }
        )
        self.order_gates = nn.ParameterDict(
            {
                str(order): nn.Parameter(torch.tensor(1e-3))
                for order in self.rows_by_order
            }
        )
        for table in self.tables.values():
            nn.init.normal_(table.weight, std=memory_width**-0.5)
        with torch.no_grad():
            for projection in self.projections.values():
                projection.weight.mul_(self.initialization_gain)

    def forward(self, addresses: Mapping[int, Tensor] | None) -> Tensor | None:
        if not addresses:
            return None
        output: Tensor | None = None
        for order in sorted(self.rows_by_order):
            raw = addresses.get(order)
            if raw is None:
                continue
            if raw.ndim != 3 or raw.shape[-1] != self.heads:
                raise ValueError(
                    f"Hash addresses for order {order} must be [B,T,{self.heads}]"
                )
            valid = raw >= 0
            safe = raw.clamp(min=0, max=self.rows_by_order[order] - 1)
            values = self.tables[str(order)](safe)
            values = values * valid.unsqueeze(-1)
            denominator = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            pooled = values.sum(dim=-2) / denominator
            projected = self.projections[str(order)](pooled)
            gated = torch.tanh(self.order_gates[str(order)]) * projected
            output = gated if output is None else output + gated
        return output


class SparseMemoryFusion(nn.Module):
    """Fuse addressed rows while keeping definitions in their native space.

    The surface index chooses rows.  JTD maps layer-8, layer-24, and live
    states into the layer-2 reference frame.  Every retrieved definition sense
    remains present as its own point.  A continuous inverse-distance field
    supplies locality weights on each visit, so the recurrent blocks receive
    concept geometry without a block-owned sense selector or episode latch.
    """

    def __init__(
        self,
        model_width: int,
        *,
        memory_width: int,
        hash_rows_by_order: Mapping[int, int],
        hash_heads: int,
        hash_memory_width: int,
        residual_initialization_gain: float = 1.0,
        definition_harmonic_exponent: float = 2.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.model_width = int(model_width)
        self.memory_width = int(memory_width)
        self.definition_harmonic_exponent = float(definition_harmonic_exponent)
        if self.definition_harmonic_exponent <= 0:
            raise ValueError("definition_harmonic_exponent must be positive")
        self.epsilon = float(epsilon)
        self.joint_transfer = JointTransferDomain(
            model_width,
            memory_width=memory_width,
        )
        self.hash_memory = TrainableHashEngram(
            model_width,
            rows_by_order=hash_rows_by_order,
            heads=hash_heads,
            memory_width=hash_memory_width,
            initialization_gain=residual_initialization_gain,
        )

        # Index 0 is physical block 1; index 1 is physical block 2.  Near-zero
        # values preserve a quiet start while keeping every path trainable.
        self.phrase8_gates = nn.Parameter(torch.full((2,), 1e-3))
        self.phrase24_gates = nn.Parameter(torch.full((2,), 1e-3))
        self.definition_gates = nn.Parameter(torch.full((2,), 1e-3))
        self.hash_gates = nn.Parameter(torch.full((2,), 1e-3))
        self.initial_phrase8_gate = nn.Parameter(torch.tensor(1e-3))
        self.initial_hash_gate = nn.Parameter(torch.tensor(1e-3))

    def _validate_phrase(self, values: Tensor, hidden: Tensor) -> None:
        if values.shape != (*hidden.shape[:2], self.memory_width):
            raise ValueError(
                "Phrase memory must be "
                f"[B,T,{self.memory_width}], found {tuple(values.shape)}"
            )

    def _phrase_update(
        self,
        values: Tensor,
        hidden: Tensor,
        *,
        source: str,
        mask: Tensor | None,
    ) -> Tensor:
        self._validate_phrase(values, hidden)
        joint = self.joint_transfer.source_to_joint(values, source)  # type: ignore[arg-type]
        update = self.joint_transfer.movement_to_live(joint)
        if mask is not None:
            if mask.shape != hidden.shape[:2]:
                raise ValueError("phrase_mask must be [B,T]")
            update = update * mask.unsqueeze(-1)
        return update

    def _definition_field(
        self,
        hidden: Tensor,
        values: Tensor | None,
        mask: Tensor | None,
        sense_rows: Tensor | None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        if values is None:
            return None, None, None, None
        if values.ndim != 4 or values.shape[:2] != hidden.shape[:2]:
            raise ValueError("Definitions must be [B,T,S,memory_width]")
        if values.shape[-1] != self.memory_width:
            raise ValueError("Definition width differs from configured memory width")
        if mask is None:
            mask = torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
        if mask.shape != values.shape[:-1]:
            raise ValueError("definition_mask must be [B,T,S]")
        mask = mask.to(device=values.device, dtype=torch.bool)
        if sense_rows is not None:
            if sense_rows.shape != values.shape[:-1]:
                raise ValueError("definition_sense_rows must be [B,T,S]")
            sense_rows = sense_rows.to(device=values.device, dtype=torch.long)

        anchors = self.joint_transfer.definitions_to_joint(values)
        query = self.joint_transfer.source_to_joint(hidden, "live")
        displacement = anchors - query.unsqueeze(-2)
        squared_distances = (
            displacement.square().mean(dim=-1) + self.epsilon**2
        )
        inverse_distance_mass = squared_distances.pow(
            -0.5 * self.definition_harmonic_exponent
        )
        inverse_distance_mass = inverse_distance_mass * mask
        denominator = inverse_distance_mass.sum(dim=-1, keepdim=True)
        weights = inverse_distance_mass / (denominator + self.epsilon)
        active = mask.any(dim=-1)
        centroid = (weights.unsqueeze(-1) * anchors).sum(dim=-2)
        movement = (centroid - query) * active.unsqueeze(-1)
        update = self.joint_transfer.movement_to_live(movement)
        return update, weights, squared_distances, sense_rows

    def initial_update(
        self,
        hidden: Tensor,
        payloads: MemoryPayloads | None,
    ) -> Tensor:
        if payloads is None:
            return torch.zeros_like(hidden)
        update = torch.zeros_like(hidden)
        if payloads.phrase_layer8 is not None:
            phrase = self._phrase_update(
                payloads.phrase_layer8,
                hidden,
                source="layer8",
                mask=payloads.phrase_mask,
            )
            update = update + torch.tanh(self.initial_phrase8_gate) * phrase
        hashed = self.hash_memory(payloads.hash_addresses)
        if hashed is not None:
            update = update + torch.tanh(self.initial_hash_gate) * hashed
        return update

    def forward(
        self,
        hidden: Tensor,
        payloads: MemoryPayloads | None,
        *,
        block_index: int,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, MemoryFusionStats]:
        if block_index not in {0, 1}:
            raise ValueError("block_index must be 0 or 1")
        if payloads is None:
            update = torch.zeros_like(hidden)
            if return_stats:
                zero = hidden.new_tensor(0.0)
                return update, MemoryFusionStats(
                    phrase8_gate=zero,
                    phrase24_gate=zero,
                    definition_gate=zero,
                    hash_gate=zero,
                    definition_weights=None,
                    definition_squared_distances=None,
                    definition_sense_rows=None,
                    active_definition_count=None,
                )
            return update

        update = torch.zeros_like(hidden)
        if payloads.phrase_layer8 is not None:
            phrase8 = self._phrase_update(
                payloads.phrase_layer8,
                hidden,
                source="layer8",
                mask=payloads.phrase_mask,
            )
            update = update + torch.tanh(self.phrase8_gates[block_index]) * phrase8

        if payloads.phrase_layer24 is not None and block_index == 1:
            phrase24 = self._phrase_update(
                payloads.phrase_layer24,
                hidden,
                source="layer24",
                mask=payloads.phrase_mask,
            )
            update = (
                update
                + torch.tanh(self.phrase24_gates[block_index]) * phrase24
            )

        definition_update, weights, distances, sense_rows = self._definition_field(
            hidden,
            payloads.definitions,
            payloads.definition_mask,
            payloads.definition_sense_rows,
        )
        if definition_update is not None:
            update = (
                update
                + torch.tanh(self.definition_gates[block_index])
                * definition_update
            )

        hashed = self.hash_memory(payloads.hash_addresses)
        if hashed is not None:
            update = update + torch.tanh(self.hash_gates[block_index]) * hashed

        if return_stats:
            active_count = (
                None
                if payloads.definition_mask is None
                else payloads.definition_mask.to(torch.long).sum(dim=-1)
            )
            return update, MemoryFusionStats(
                phrase8_gate=torch.tanh(self.phrase8_gates[block_index]),
                phrase24_gate=torch.tanh(self.phrase24_gates[block_index]),
                definition_gate=torch.tanh(self.definition_gates[block_index]),
                hash_gate=torch.tanh(self.hash_gates[block_index]),
                definition_weights=weights,
                definition_squared_distances=distances,
                definition_sense_rows=sense_rows,
                active_definition_count=active_count,
            )
        return update
