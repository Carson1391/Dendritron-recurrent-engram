"""Configuration for the runnable Dendritron language core.

The production width is 2,048 so the live state can receive the completed
Qwen donor rows without an information-losing bottleneck.  Tiny widths remain
available for structural smoke tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


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

    universal_rank: int = 16
    universal_top_k: int = 4
    universal_bootstrap_seed: int | None = None

    skill_count: int = 21
    skill_rank: int = 8
    skill_top_k: int = 4

    expert_count: int = 11
    expert_hidden_width: int = 512
    expert_branches: int = 2
    expert_top_k: int = 2

    use_lngram: bool = True
    lngram_bits_per_route: int = 4
    lngram_orders: tuple[int, ...] = (2, 3)
    lngram_route_memory_width: int = 4

    hash_orders: tuple[int, ...] = (2, 3)
    hash_heads: int = 4
    hash_table_rows: tuple[int, ...] = (65_536, 262_144)
    hash_memory_width: int = 88

    vocabulary_chunk_size: int = 4096

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
            "skill_count": self.skill_count,
            "skill_rank": self.skill_rank,
            "skill_top_k": self.skill_top_k,
            "expert_count": self.expert_count,
            "expert_hidden_width": self.expert_hidden_width,
            "expert_branches": self.expert_branches,
            "expert_top_k": self.expert_top_k,
            "hash_heads": self.hash_heads,
            "hash_memory_width": self.hash_memory_width,
            "vocabulary_chunk_size": self.vocabulary_chunk_size,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.universal_rank < 0:
            raise ValueError("universal_rank must be nonnegative")
        if self.universal_rank and self.universal_top_k < 1:
            raise ValueError("universal_top_k must be positive when directions exist")
        if self.skill_top_k > self.skill_count:
            raise ValueError("skill_top_k cannot exceed skill_count")
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

    @property
    def block_equivalent_depth(self) -> int:
        """Unrolled block depth N = K R for the locked K=2 core."""

        return 2 * self.loop_rounds

    @property
    def deep_loop_alpha(self) -> float:
        """Post-RMSNorm skip gain applied on every residual visit."""

        return float((2.0 * self.block_equivalent_depth) ** self.deep_loop_exponent)

    @property
    def deep_loop_beta(self) -> float:
        """One-time initialization gain for residual-branch matrices."""

        return float((8.0 * self.block_equivalent_depth) ** (-self.deep_loop_exponent))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["hash_rows_by_order"] = self.hash_rows_by_order
        result["block_equivalent_depth"] = self.block_equivalent_depth
        result["deep_loop_alpha"] = self.deep_loop_alpha
        result["deep_loop_beta"] = self.deep_loop_beta
        return result

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "DendritronConfig":
        """Reconstruct a config while ignoring derived manifest fields."""

        accepted = {field.name for field in fields(cls)}
        values = {key: value for key, value in record.items() if key in accepted}
        for key in (
            "lngram_orders",
            "hash_orders",
            "hash_table_rows",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def tiny_smoke_config(vocab_size: int) -> DendritronConfig:
    """A small, structurally identical configuration for a local smoke run."""

    return DendritronConfig(
        vocab_size=vocab_size,
        model_width=64,
        memory_width=64,
        max_sequence_length=64,
        loop_rounds=2,
        context_window=16,
        context_top_k=8,
        universal_rank=4,
        universal_top_k=2,
        universal_bootstrap_seed=7,
        skill_count=7,
        skill_rank=4,
        skill_top_k=2,
        expert_count=9,
        expert_hidden_width=512,
        expert_branches=1,
        expert_top_k=2,
        lngram_bits_per_route=4,
        lngram_orders=(2, 3),
        lngram_route_memory_width=4,
        hash_orders=(2, 3),
        hash_heads=2,
        hash_table_rows=(256, 1024),
        hash_memory_width=28,
    )
