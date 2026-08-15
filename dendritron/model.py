"""Runnable Dendritron language model assembled from the completed subsystems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.model requires PyTorch. Install torch>=2.7."
    ) from error

from .capacity import SparseCapacityLedger
from .config import DendritronConfig
from .memory_fusion import MemoryPayloads, SparseMemoryFusion
from .output_geometry import GeometricVocabularyHead
from .recurrent_core import RecurrentCoreStats, TwoBlockRecurrentCore


@dataclass(frozen=True)
class DendritronOutput:
    logits: Tensor
    hidden: Tensor
    recurrent_stats: RecurrentCoreStats | None = None


def _sinusoidal_geometry(length: int, width: int) -> Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / max(width, 1))
    )
    geometry = torch.zeros(length, width, dtype=torch.float32)
    geometry[:, 0::2] = torch.sin(positions * frequencies)
    if width > 1:
        geometry[:, 1::2] = torch.cos(
            positions * frequencies[: geometry[:, 1::2].shape[1]]
        )
    return geometry


def _parameter_count(
    module: nn.Module,
    *,
    trainable_only: bool = False,
    matrices_only: bool = False,
) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
        if not matrices_only or parameter.ndim >= 2
    )


class DendritronLM(nn.Module):
    """Two-block, memory-grafted, softmax-free autoregressive model."""

    def __init__(self, config: DendritronConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.model_width)
        nn.init.normal_(
            self.token_embeddings.weight,
            std=config.model_width**-0.5,
        )
        self.register_buffer(
            "position_geometry",
            _sinusoidal_geometry(
                config.max_sequence_length,
                config.model_width,
            ),
            persistent=True,
        )
        self.memory_fusion = SparseMemoryFusion(
            config.model_width,
            memory_width=config.memory_width,
            hash_rows_by_order=config.hash_rows_by_order,
            hash_heads=config.hash_heads,
            hash_memory_width=config.hash_memory_width,
            definition_harmonic_exponent=config.harmax_exponent,
            epsilon=config.geometric_epsilon,
        )
        self.core = TwoBlockRecurrentCore(
            config,
            memory_fusion=self.memory_fusion,
        )
        self.vocabulary_head = GeometricVocabularyHead(
            config.model_width,
            epsilon=config.geometric_epsilon,
        )

    @torch.no_grad()
    def load_token_embeddings(self, values: Tensor) -> None:
        if tuple(values.shape) != tuple(self.token_embeddings.weight.shape):
            raise ValueError(
                "Token embedding shape mismatch: "
                f"expected {tuple(self.token_embeddings.weight.shape)}, "
                f"found {tuple(values.shape)}"
            )
        self.token_embeddings.weight.copy_(
            values.to(
                device=self.token_embeddings.weight.device,
                dtype=self.token_embeddings.weight.dtype,
            )
        )

    @torch.no_grad()
    def load_token_embedding_artifact(self, path: str | Path) -> None:
        """Load the exact Qwen input-symbol table exported offline.

        The artifact contains one row for every raw Qwen tokenizer ID.  This
        includes punctuation, whitespace, subword pieces, and special symbols;
        complete words still use the separate dictionary-sense lookup.
        """

        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover - runtime dependency
            raise RuntimeError(
                "Token embedding artifacts require safetensors>=0.5"
            ) from error
        artifact = Path(path)
        with safe_open(str(artifact), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"token_embeddings"}:
                raise ValueError(
                    "Token embedding artifact must contain token_embeddings"
                )
            values = handle.get_tensor("token_embeddings")
        self.load_token_embeddings(values)

    @torch.no_grad()
    def load_skill_factors(
        self,
        shared_alpha: Tensor,
        shared_beta: Tensor,
        skill_coeffs: Tensor,
        private_alpha: Tensor,
        private_beta: Tensor,
        anchors: Tensor,
        *,
        slot_populated: Tensor | None = None,
        private_ranks: Tensor | None = None,
    ) -> None:
        """Load frozen shared basis, per-skill coefficients, private LoRAs, and anchors.

        Args:
            shared_alpha:   [block_count, K, D] shared input factors a_{b,j} (frozen).
            shared_beta:    [block_count, D, K] shared output factors b_{b,j} (frozen).
            skill_coeffs:   [S, K] per-skill coefficients q_{s,j}, shared across blocks (frozen).
            private_alpha:  [block_count, S, R_max, D] private input factors (frozen, padded).
            private_beta:   [block_count, S, D, R_max] private output factors (frozen, padded).
            anchors:        [S, D] routing anchors (frozen in production).
            slot_populated: [S] bool, True for populated skill slots.  If None, all populated.
            private_ranks:  [S] int, actual rank per slot (<= R_max).  If None, all at R_max.
        """
        self.core.skill_expert.skills.load_skill_factors(
            shared_alpha, shared_beta, skill_coeffs,
            private_alpha, private_beta, anchors,
            slot_populated=slot_populated,
            private_ranks=private_ranks,
        )

    @torch.no_grad()
    def load_expert_registry(
        self,
        anchors: Tensor,
        adjacency: Tensor,
    ) -> None:
        """Load expert anchors and skill-expert adjacency from evidence.

        Args:
            anchors: [expert_count, width] from clustered residual trajectories.
            adjacency: [max_skill_slots, expert_count] bool mask.
        """
        self.core.skill_expert.experts.load_expert_registry(anchors, adjacency)

    @torch.no_grad()
    def load_joint_transfer_checkpoint(self, path: str) -> None:
        """Load fitted source maps into the layer-2 definition frame."""

        record = torch.load(path, map_location="cpu", weights_only=True)
        transfer = self.memory_fusion.joint_transfer
        for source in ("layer8", "layer24", "live"):
            key = f"{source}_to_joint"
            if key not in record:
                raise ValueError(f"JTD checkpoint is missing {key}")
            transfer.load_projection(source, record[key])
        joint_to_live = record.get("joint_to_live")
        if joint_to_live is None:
            raise ValueError("JTD checkpoint is missing joint_to_live")
        if tuple(joint_to_live.shape) != tuple(transfer.joint_to_live.weight.shape):
            raise ValueError("joint_to_live projection shape mismatch")
        transfer.joint_to_live.weight.copy_(
            joint_to_live.to(
                device=transfer.joint_to_live.weight.device,
                dtype=transfer.joint_to_live.weight.dtype,
            )
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        memory_payloads: MemoryPayloads | None = None,
        rounds: int | None = None,
        adaptive_threshold: float | None = None,
        minimum_rounds: int = 1,
        return_stats: bool = False,
    ) -> DendritronOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        if input_ids.shape[1] > self.config.max_sequence_length:
            raise ValueError("Input exceeds configured maximum sequence length")
        if input_ids.numel() and (
            int(input_ids.min()) < 0
            or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("input_ids contain an out-of-range token")

        length = input_ids.shape[1]
        hidden = self.token_embeddings(input_ids)
        hidden = hidden + self.position_geometry[:length].to(hidden).unsqueeze(0)
        hidden = hidden + self.memory_fusion.initial_update(hidden, memory_payloads)

        if return_stats:
            hidden, recurrent_stats = self.core(
                hidden,
                memory_payloads=memory_payloads,
                rounds=rounds,
                adaptive_threshold=adaptive_threshold,
                minimum_rounds=minimum_rounds,
                return_stats=True,
            )
        else:
            hidden = self.core(
                hidden,
                memory_payloads=memory_payloads,
                rounds=rounds,
                adaptive_threshold=adaptive_threshold,
                minimum_rounds=minimum_rounds,
                return_stats=False,
            )
            recurrent_stats = None
        logits = self.vocabulary_head(hidden, self.token_embeddings.weight)
        return DendritronOutput(
            logits=logits,
            hidden=hidden,
            recurrent_stats=recurrent_stats,
        )

    def capacity_ledger(
        self,
        *,
        external_memory_parameters: int = 0,
    ) -> SparseCapacityLedger:
        if external_memory_parameters < 0:
            raise ValueError("external_memory_parameters must be nonnegative")
        hash_parameters = sum(
            table.weight.numel()
            for table in self.memory_fusion.hash_memory.tables.values()
        )
        lngram_parameters = 0
        if self.core.lngram is not None:
            for module in self.core.lngram:
                lngram_parameters += sum(
                    table.numel() for table in module.tables.values()
                )
        memory_parameters = (
            int(external_memory_parameters)
            + hash_parameters
            + lngram_parameters
        )
        # The fixed sparse-capacity ratio counts primary table and matrix
        # coefficients. Biases, gates, normalizers, and routing vectors are
        # control metadata and remain in the shared-core report.
        compute_parameters = _parameter_count(
            self.core.skill_expert,
            matrices_only=True,
        )
        # Expert routing anchors are 2D tensors but are control metadata,
        # not primary compute matrices. Subtract them from the compute count
        # so the 25/75 split reflects only skill factors + expert branch weights.
        expert_anchor_params = self.core.skill_expert.experts.anchors.numel()
        compute_parameters -= expert_anchor_params
        total_parameters = _parameter_count(self)
        locally_counted = hash_parameters + lngram_parameters + compute_parameters + expert_anchor_params
        shared_core_parameters = max(total_parameters - locally_counted, 0)
        return SparseCapacityLedger(
            memory_parameters=memory_parameters,
            compute_parameters=compute_parameters,
            shared_core_parameters=shared_core_parameters,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        rounds: int | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [B,T]")
        generated = input_ids
        for _ in range(int(max_new_tokens)):
            window = generated[:, -self.config.max_sequence_length :]
            output = self(window, rounds=rounds)
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_id is not None and bool((next_token == int(eos_id)).all()):
                break
        return generated
