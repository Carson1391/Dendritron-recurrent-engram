"""Pure-Python/NumPy LNGram address reference implementation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def pack_route_bits(bits: np.ndarray) -> np.ndarray:
    """Pack [..., M] binary values into integer route symbols."""
    values = np.asarray(bits, dtype=np.int64)
    if values.ndim < 1:
        raise ValueError("bits must have a final bit dimension")
    bit_count = values.shape[-1]
    powers = np.left_shift(np.int64(1), np.arange(bit_count, dtype=np.int64))
    return np.sum(values * powers, axis=-1, dtype=np.int64)


def lngram_addresses(
    symbols: np.ndarray,
    *,
    order: int,
    alphabet_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact route-partitioned n-gram addresses and a validity mask.

    ``symbols`` is shaped ``[..., T, R]``. Returned addresses have the same
    shape. Positions before a complete n-gram are filled with zero and marked
    invalid.
    """
    values = np.asarray(symbols, dtype=np.int64)
    if values.ndim < 2:
        raise ValueError("symbols must be shaped [..., sequence, routes]")
    if order < 1:
        raise ValueError("order must be positive")
    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least two")
    if np.any(values < 0) or np.any(values >= alphabet_size):
        raise ValueError("symbols contain values outside the alphabet")

    length = values.shape[-2]
    routes = values.shape[-1]
    addresses = np.zeros_like(values)
    valid = np.zeros_like(values, dtype=bool)
    route_offsets = (
        np.arange(routes, dtype=np.int64) * (alphabet_size**order)
    )
    for end in range(order - 1, length):
        address = np.broadcast_to(route_offsets, values.shape[:-2] + (routes,)).copy()
        start = end - order + 1
        for local_position in range(order):
            address += values[..., start + local_position, :] * (
                alphabet_size**local_position
            )
        addresses[..., end, :] = address
        valid[..., end, :] = True
    return addresses, valid


def table_rows(route_count: int, alphabet_size: int, order: int) -> int:
    return int(route_count * alphabet_size**order)


def address_sequence(
    route_symbols: Iterable[int],
    *,
    route: int,
    alphabet_size: int,
) -> int:
    symbols = [int(value) for value in route_symbols]
    return route * alphabet_size ** len(symbols) + sum(
        symbol * alphabet_size**index for index, symbol in enumerate(symbols)
    )
