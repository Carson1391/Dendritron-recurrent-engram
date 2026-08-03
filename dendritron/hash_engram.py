"""Deterministic token-ID hashing for the trainable Engram fallback.

This path is distinct from both frozen donor Engrams and LNGram:

* frozen donor Engrams use collision-checked canonical Qwen-ID addresses;
* Hash Engram maps uncovered canonical Qwen-ID suffixes into trainable rows;
* LNGram discretizes live hidden states into a separate latent address space.

The reference hash uses arithmetic that stays within signed 64-bit range so a
later Torch implementation can reproduce the same addresses on CPU and GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


HASH_SCHEMA_VERSION = 1
HASH_MODULUS = 2_147_483_647
HASH_MULTIPLIER = 1_000_003
DEFAULT_HASH_ORDERS = (2, 3)
DEFAULT_HASH_HEADS = 4
DEFAULT_TABLE_ROWS = {2: 1_048_576, 3: 4_194_304}


def hash_ngram_ids(
    token_ids: Sequence[int],
    *,
    order: int,
    head: int,
    table_rows: int,
) -> int:
    """Map one exact token suffix to a deterministic trainable-table row."""
    if order < 1:
        raise ValueError("order must be positive")
    if len(token_ids) != order:
        raise ValueError(
            f"Expected {order} token IDs for the hash key, found {len(token_ids)}"
        )
    if head < 0:
        raise ValueError("head must be nonnegative")
    if table_rows < 1:
        raise ValueError("table_rows must be positive")

    state = (1_000_033 + order * 10_007 + head * 1_000_037) % HASH_MODULUS
    for position, raw_token_id in enumerate(token_ids):
        token_id = int(raw_token_id)
        if token_id < 0:
            raise ValueError("token IDs must be nonnegative")
        state = (
            state * HASH_MULTIPLIER
            + token_id
            + (position + 1) * 97
        ) % HASH_MODULUS
    return state % table_rows


@dataclass(frozen=True)
class HashEngramAddresses:
    end_position: int
    by_order: dict[int, tuple[int, ...]]


class HashEngramAddressor:
    """Multi-head Engram address generator for exact-memory misses."""

    def __init__(
        self,
        *,
        orders: Sequence[int] = DEFAULT_HASH_ORDERS,
        heads: int = DEFAULT_HASH_HEADS,
        table_rows: Mapping[int, int] = DEFAULT_TABLE_ROWS,
    ) -> None:
        normalized_orders = tuple(sorted({int(value) for value in orders}))
        if not normalized_orders or min(normalized_orders) < 1:
            raise ValueError("orders must contain positive integers")
        if heads < 1:
            raise ValueError("heads must be positive")
        missing = set(normalized_orders) - {int(key) for key in table_rows}
        if missing:
            raise ValueError(f"Missing table sizes for orders {sorted(missing)}")
        self.orders = normalized_orders
        self.heads = int(heads)
        self.table_rows = {
            order: int(table_rows[order])
            for order in normalized_orders
        }

    def addresses_ending_at(
        self,
        token_ids: Sequence[int],
        end_position: int,
    ) -> HashEngramAddresses:
        if not 0 <= end_position < len(token_ids):
            raise IndexError("end_position falls outside token_ids")
        result: dict[int, tuple[int, ...]] = {}
        for order in self.orders:
            start = end_position - order + 1
            if start < 0:
                continue
            suffix = token_ids[start : end_position + 1]
            result[order] = tuple(
                hash_ngram_ids(
                    suffix,
                    order=order,
                    head=head,
                    table_rows=self.table_rows[order],
                )
                for head in range(self.heads)
            )
        return HashEngramAddresses(
            end_position=end_position,
            by_order=result,
        )

    def manifest_record(self) -> dict[str, object]:
        return {
            "schema_version": HASH_SCHEMA_VERSION,
            "address_source": "canonical_projection_of_qwen_token_ids",
            "orders": list(self.orders),
            "heads": self.heads,
            "table_rows": {
                str(order): self.table_rows[order]
                for order in self.orders
            },
            "hash": {
                "name": "bounded_polynomial_v1",
                "modulus": HASH_MODULUS,
                "multiplier": HASH_MULTIPLIER,
            },
            "activation": "exact_donor_engram_miss",
        }
