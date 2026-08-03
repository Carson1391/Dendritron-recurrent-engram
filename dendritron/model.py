"""Runnable Dendritron language model assembled from the completed subsystems."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
            residual_initialization_gain=config.deep_loop_beta,
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
            vocabulary_chunk_size=config.vocabulary_chunk_size,
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["DendritronLM", dict]:
        """Load a Dendritron checkpoint and its accompanying metadata."""

        record = torch.load(path, map_location=map_location, weights_only=True)
        if "model" not in record or "config" not in record:
            raise ValueError("Checkpoint must contain model and config records")
        config = DendritronConfig.from_dict(record["config"])
        model = cls(config).to(map_location)
        model.load_state_dict(record["model"])
        return model, record

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
    def load_universal_direction_files(
        self,
        block_files: tuple[str, str],
    ) -> None:
        """Load HOSVD input/output directions for physical blocks 1 and 2.

        Files are the ``.pt`` records produced by
        ``stage4_subspace/qwen_lora_universal_directions.py``.  ``U_in`` becomes
        the input-side alpha direction and ``U_out`` the output-side beta
        direction.  The configured spectral rank is retained.
        """

        rank = self.config.universal_rank
        if rank == 0:
            raise ValueError("The model was configured without universal directions")
        alpha = []
        beta = []
        for path in block_files:
            record = torch.load(path, map_location="cpu", weights_only=True)
            input_directions = record["U_in"]
            output_directions = record["U_out"]
            if input_directions.shape[0] != self.config.model_width:
                raise ValueError(f"U_in width mismatch in {path}")
            if output_directions.shape[0] != self.config.model_width:
                raise ValueError(f"U_out width mismatch in {path}")
            if min(input_directions.shape[1], output_directions.shape[1]) < rank:
                raise ValueError(f"Direction file {path} contains fewer than {rank} axes")
            alpha.append(input_directions[:, :rank].T.contiguous())
            beta.append(output_directions[:, :rank].T.contiguous())
        self.core.skill_expert.universal.load_directions(
            torch.stack(alpha),
            torch.stack(beta),
        )

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
        total_parameters = _parameter_count(self)
        locally_counted = hash_parameters + lngram_parameters + compute_parameters
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
