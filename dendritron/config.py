"""Configuration for the runnable Dendritron language core.

The production width is 2,048 so the live state can receive the completed
Qwen donor rows without an information-losing bottleneck.  The smoke
configuration shares this geometry and varies only workload sizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DendritronConfig:
    vocab_size: int
    model_width: int = 2048
    memory_width: int = 2048
    max_sequence_length: int = 1024
    loop_rounds: int = 2
    context_window: int = 128
    context_top_k: int = 32
    harmax_exponent: float = 2.0

    max_skill_slots: int = 32
    shared_basis_count: int = 16
    max_private_lora_rank: int = 4
    skill_top_k: int = 4
    init_mode: str = "smoke"

    expert_count: int = 11
    expert_hidden_width: int = 512
    expert_branches: int = 2
    expert_top_k: int = 2

    # Typed deductive branch configuration
    deductive_branches_per_block: int = 0
    deductive_max_premises: int = 8
    deductive_max_contradictions: int = 4

    use_lngram: bool = True
    lngram_bits_per_route: int = 4
    lngram_orders: tuple[int, ...] = (2, 3)
    lngram_route_memory_width: int = 4

    hash_orders: tuple[int, ...] = (2, 3)
    hash_heads: int = 4
    hash_table_rows: tuple[int, ...] = (65_536, 262_144)
    hash_memory_width: int = 88

    memory_fraction: float = 0.25
    deep_loop_exponent: float = 0.5
    residual_epsilon: float = 1e-6
    geometric_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "model_width": self.model_width,
            "memory_width": self.memory_width,
            "max_sequence_length": self.max_sequence_length,
            "loop_rounds": self.loop_rounds,
            "context_window": self.context_window,
            "context_top_k": self.context_top_k,
            "max_skill_slots": self.max_skill_slots,
            "skill_top_k": self.skill_top_k,
            "expert_count": self.expert_count,
            "expert_hidden_width": self.expert_hidden_width,
            "expert_branches": self.expert_branches,
            "expert_top_k": self.expert_top_k,
            "hash_heads": self.hash_heads,
            "hash_memory_width": self.hash_memory_width,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.init_mode not in ("smoke", "production"):
            raise ValueError("init_mode must be 'smoke' or 'production'")
        if self.shared_basis_count < 1:
            raise ValueError("shared_basis_count must be positive")
        if self.max_private_lora_rank < 1:
            raise ValueError("max_private_lora_rank must be positive")
        if self.skill_top_k > self.max_skill_slots:
            raise ValueError("skill_top_k cannot exceed max_skill_slots")
        if self.expert_top_k > self.expert_count:
            raise ValueError("expert_top_k cannot exceed expert_count")
        if self.context_window > self.max_sequence_length:
            raise ValueError("context_window cannot exceed max_sequence_length")
        if self.context_top_k > self.context_window:
            raise ValueError("context_top_k cannot exceed context_window")
        if self.harmax_exponent <= 0:
            raise ValueError("harmax_exponent must be positive")
        if self.model_width % self.lngram_bits_per_route:
            raise ValueError(
                "model_width must be divisible by lngram_bits_per_route"
            )
        if len(self.hash_orders) != len(self.hash_table_rows):
            raise ValueError(
                "hash_orders and hash_table_rows must contain the same number of items"
            )
        if any(order < 1 for order in self.hash_orders):
            raise ValueError("hash orders must be positive")
        if any(rows < 1 for rows in self.hash_table_rows):
            raise ValueError("hash table row counts must be positive")
        if abs(self.memory_fraction - 0.25) > 1e-12:
            raise ValueError("Dendritron fixes conditional memory capacity at 25%")
        if self.deep_loop_exponent not in {0.25, 0.5}:
            raise ValueError("deep_loop_exponent must be 0.25 or 0.5")

    @property
    def hash_rows_by_order(self) -> dict[int, int]:
        return dict(zip(self.hash_orders, self.hash_table_rows, strict=True))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["hash_rows_by_order"] = self.hash_rows_by_order
        return result


def tiny_smoke_config(vocab_size: int) -> DendritronConfig:
    """Smoke configuration sharing production geometry with reduced workload.

    Only workload sizes differ from production: vocabulary, sequence length,
    context window, and hash table fixture rows.  All model geometry (width,
    rank, skill/expert counts, hash heads, hash memory width, LNGram
    configuration) matches production so smoke tests exercise the real
    parameter shapes.
    """

    return DendritronConfig(
        vocab_size=vocab_size,
        max_sequence_length=64,
        context_window=16,
        context_top_k=8,
        init_mode="smoke",
        hash_table_rows=(256, 1024),
    )
