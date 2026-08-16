"""Frozen address-to-sense-row lookup table for LNGram definition routing.

Each address in the route-partitioned n-gram table maps to
``senses_per_address`` sense row indices into the frozen definition bank.
The mask marks which sense slots are populated, and evidence stores
the target attraction strength for each sense.

All buffers are frozen (non-trainable).  Population happens through
``populate()`` using verified state-to-sense examples from JTD-aligned
coordinates.
"""

from __future__ import annotations

from typing import Iterable

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.address_table requires PyTorch. Install torch>=2.7."
    ) from error


class AddressRecordTable(nn.Module):
    """Per-address record of sense row handles, validity masks, and evidence.

    The address is the key; sense rows are the payload.  Each populated
    record contains::

        [sense_row_1, ..., sense_row_K], mask, evidence

    where ``K = senses_per_address``.  The mapping uses an explicit
    population procedure with verified state-to-sense examples.  JTD
    supplies the aligned coordinates used to train that assignment.
    """

    def __init__(
        self,
        route_count: int,
        alphabet_size: int,
        orders: Iterable[int],
        senses_per_address: int,
    ) -> None:
        super().__init__()
        self.route_count = int(route_count)
        self.alphabet_size = int(alphabet_size)
        self.orders = tuple(sorted(set(int(o) for o in orders)))
        if not self.orders or min(self.orders) < 1:
            raise ValueError("orders must contain positive integers")
        self.senses_per_address = int(senses_per_address)
        if self.senses_per_address < 1:
            raise ValueError("senses_per_address must be positive")

        for order in self.orders:
            rows = self.route_count * self.alphabet_size**order
            # sense_rows: indices into the definition bank (-1 = unpopulated)
            self.register_buffer(
                f"sense_rows_{order}",
                torch.full((rows, self.senses_per_address), -1, dtype=torch.long),
            )
            # mask: which sense slots are valid
            self.register_buffer(
                f"mask_{order}",
                torch.zeros((rows, self.senses_per_address), dtype=torch.bool),
            )
            # evidence: target attraction strength for signed HarMax (y-mass)
            self.register_buffer(
                f"evidence_{order}",
                torch.zeros((rows, self.senses_per_address), dtype=torch.float32),
            )

    @property
    def table_row_counts(self) -> dict[int, int]:
        return {
            order: self.route_count * self.alphabet_size**order
            for order in self.orders
        }

    @torch.no_grad()
    def populate(
        self,
        order: int,
        addresses: Tensor,
        sense_rows: Tensor,
        mask: Tensor,
        evidence: Tensor,
    ) -> None:
        """Fill table entries from verified state-to-sense examples.

        Args:
            order: n-gram order to populate.
            addresses: [N] long tensor of address indices.
            sense_rows: [N, senses_per_address] long tensor of bank row indices.
            mask: [N, senses_per_address] bool tensor.
            evidence: [N, senses_per_address] float tensor of target evidence.
        """
        if order not in self.orders:
            raise ValueError(f"order {order} not in configured orders {self.orders}")
        table_sense_rows = getattr(self, f"sense_rows_{order}")
        table_mask = getattr(self, f"mask_{order}")
        table_evidence = getattr(self, f"evidence_{order}")

        if addresses.ndim != 1:
            raise ValueError("addresses must be [N]")
        n = addresses.shape[0]
        if sense_rows.shape != (n, self.senses_per_address):
            raise ValueError(
                f"sense_rows must be [{n}, {self.senses_per_address}]"
            )
        if mask.shape != sense_rows.shape:
            raise ValueError("mask must match sense_rows shape")
        if evidence.shape != sense_rows.shape:
            raise ValueError("evidence must match sense_rows shape")

        if bool((addresses < 0).any() or (addresses >= table_sense_rows.shape[0]).any()):
            raise ValueError("addresses out of range")

        table_sense_rows[addresses] = sense_rows.to(
            table_sense_rows.device, dtype=torch.long
        )
        table_mask[addresses] = mask.to(table_mask.device, dtype=torch.bool)
        table_evidence[addresses] = evidence.to(
            table_evidence.device, dtype=torch.float32
        )

    def lookup(
        self,
        order: int,
        addresses: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Gather sense_rows, mask, evidence for given addresses.

        Args:
            order: n-gram order.
            addresses: [B, T, routes] long tensor.
            valid: [B, T, routes] bool tensor.

        Returns:
            sense_rows: [B, T, routes, senses_per_address] long (-1 for invalid).
            mask: [B, T, routes, senses_per_address] bool.
            evidence: [B, T, routes, senses_per_address] float.
        """
        if order not in self.orders:
            raise ValueError(f"order {order} not in configured orders {self.orders}")
        table_sense_rows = getattr(self, f"sense_rows_{order}")
        table_mask = getattr(self, f"mask_{order}")
        table_evidence = getattr(self, f"evidence_{order}")

        safe_addr = addresses.clamp(min=0, max=table_sense_rows.shape[0] - 1)
        flat = safe_addr.reshape(-1)

        gathered_sense_rows = table_sense_rows.index_select(0, flat).view(
            *addresses.shape, self.senses_per_address
        )
        gathered_mask = table_mask.index_select(0, flat).view(
            *addresses.shape, self.senses_per_address
        )
        gathered_evidence = table_evidence.index_select(0, flat).view(
            *addresses.shape, self.senses_per_address
        )

        # Zero out invalid positions
        valid_expanded = valid.unsqueeze(-1)
        gathered_sense_rows = gathered_sense_rows.masked_fill(~valid_expanded, -1)
        gathered_mask = gathered_mask & valid_expanded
        gathered_evidence = gathered_evidence * valid_expanded.float()

        return gathered_sense_rows, gathered_mask, gathered_evidence

    def is_populated(self, order: int) -> bool:
        """Check whether any entries in the order table have been populated."""
        if order not in self.orders:
            raise ValueError(f"order {order} not in configured orders {self.orders}")
        table_mask = getattr(self, f"mask_{order}")
        return bool(table_mask.any())
