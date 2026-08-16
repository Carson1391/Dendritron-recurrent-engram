"""LNGram definition memory: route in joint space, gather frozen bank, signed HarMax.

Pipeline:
    live hidden -> JTD live_to_joint -> RMSNorm -> W_q projection -> hard route symbols
    -> route-partitioned n-gram addresses -> AddressRecordTable lookup
    -> frozen definition bank gather -> signed HarMax (y-p) contraction
    -> JTD joint_to_live -> gated residual update

The counterfactual W_q backward flips each routing bit, recomputes the address,
re-gathers from the bank, and scores the difference through the signed HarMax
field.  This preserves the training gradient signal for W_q without learned
random tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

try:
    import torch
    from torch import Tensor, nn
    from torch.autograd import Function
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover
    raise ImportError(
        "dendritron.definition_lngram requires PyTorch. Install torch>=2.7."
    ) from error

from .address_table import AddressRecordTable
from .joint_transfer import JointTransferDomain


def _pack_bits(bits: Tensor) -> Tensor:
    powers = torch.bitwise_left_shift(
        torch.ones(bits.shape[-1], device=bits.device, dtype=torch.long),
        torch.arange(bits.shape[-1], device=bits.device, dtype=torch.long),
    )
    return (bits.to(torch.long) * powers).sum(dim=-1)


def _addresses(symbols: Tensor, order: int, alphabet_size: int) -> tuple[Tensor, Tensor]:
    batch, length, routes = symbols.shape
    output = torch.zeros_like(symbols)
    valid = torch.zeros_like(symbols, dtype=torch.bool)
    if length < order:
        return output, valid
    route_offsets = (
        torch.arange(routes, device=symbols.device, dtype=torch.long)
        * alphabet_size**order
    ).view(1, 1, routes)
    ends = length - order + 1
    address = route_offsets.expand(batch, ends, routes).clone()
    for local_position in range(order):
        address.add_(
            symbols[:, local_position : local_position + ends, :]
            * alphabet_size**local_position
        )
    output[:, order - 1 :, :] = address
    valid[:, order - 1 :, :] = True
    return output, valid


class _CounterfactualBankGather(Function):
    """Custom autograd: forward gathers from frozen bank via address table;
    backward computes counterfactual W_q gradient by flipping each routing bit,
    re-deriving the address, re-gathering, and scoring through signed HarMax.
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: Tensor,
        joint_hidden: Tensor,
        address_table: AddressRecordTable,
        definition_bank: Tensor,
        joint_transfer: JointTransferDomain,
        orders: tuple[int, ...],
        bits_per_route: int,
        harmonic_exponent: float,
        epsilon: float,
        surrogate_temperature: float,
        surrogate_scale: float,
    ) -> Tensor:
        route_count = logits.shape[2]
        alphabet_size = 2**bits_per_route
        symbols = _pack_bits(logits > 0)

        # Gather per-order: sense_rows, mask, evidence from address table
        # then gather definition vectors from frozen bank
        all_anchors = []  # [B, T, routes*senses, D] per order
        all_evidence = []
        all_valid = []
        all_addresses = []
        all_valid_addr = []

        for order in orders:
            addresses, addr_valid = _addresses(symbols, order, alphabet_size)
            sense_rows, sense_mask, evidence = address_table.lookup(
                order, addresses, addr_valid
            )
            # sense_rows: [B, T, routes, senses], sense_mask: same, evidence: same
            batch, length, routes, senses = sense_rows.shape
            bank_width = definition_bank.shape[1]

            # Gather from frozen bank: flatten sense_rows, index_select, reshape
            valid_rows = sense_rows >= 0
            safe_rows = sense_rows.clamp(min=0, max=definition_bank.shape[0] - 1)
            gathered = definition_bank.index_select(
                0, safe_rows.reshape(-1)
            ).view(batch, length, routes, senses, bank_width)

            # Effective mask: address valid AND sense slot populated
            effective_valid = sense_mask & valid_rows & addr_valid.unsqueeze(-1)
            gathered = gathered * effective_valid.unsqueeze(-1).float()

            # Flatten routes*senses into the anchor pool dimension
            anchors = gathered.view(batch, length, routes * senses, bank_width)
            ev = evidence.view(batch, length, routes * senses)
            vmask = effective_valid.view(batch, length, routes * senses)

            all_anchors.append(anchors)
            all_evidence.append(ev)
            all_valid.append(vmask)
            all_addresses.append(addresses)
            all_valid_addr.append(addr_valid)

        ctx.save_for_backward(logits, joint_hidden, definition_bank)
        ctx.symbols = symbols
        ctx.address_table = address_table
        ctx.joint_transfer = joint_transfer
        ctx.orders = orders
        ctx.bits_per_route = bits_per_route
        ctx.alphabet_size = alphabet_size
        ctx.harmonic_exponent = harmonic_exponent
        ctx.epsilon = epsilon
        ctx.surrogate_temperature = surrogate_temperature
        ctx.surrogate_scale = surrogate_scale
        ctx.all_addresses = all_addresses
        ctx.all_valid_addr = all_valid_addr

        # Store forward gather results for backward reuse
        ctx.all_anchors = all_anchors
        ctx.all_evidence = all_evidence
        ctx.all_valid = all_valid

        # Compute signed HarMax movement in joint space
        movement = _signed_harmax_movement(
            joint_hidden,
            all_anchors,
            all_evidence,
            all_valid,
            harmonic_exponent,
            epsilon,
        )
        return movement

    @staticmethod
    def backward(ctx: Any, grad_movement: Tensor):
        logits, joint_hidden, definition_bank = ctx.saved_tensors
        symbols = ctx.symbols
        orders = ctx.orders
        bit_count = ctx.bits_per_route
        alphabet_size = ctx.alphabet_size
        address_table = ctx.address_table
        harmonic_exponent = ctx.harmonic_exponent
        epsilon = ctx.epsilon
        surrogate_temperature = ctx.surrogate_temperature
        surrogate_scale = ctx.surrogate_scale

        batch, length, routes, bits = logits.shape
        grad_logits = torch.zeros_like(logits)

        # For each order, for each position, for each bit in each route:
        # flip the bit -> new symbol -> new address -> new sense rows
        # -> new gather -> new HarMax movement -> score difference
        for order_idx, order in enumerate(orders):
            addresses = ctx.all_addresses[order_idx]
            addr_valid = ctx.all_valid_addr[order_idx]
            forward_anchors = ctx.all_anchors[order_idx]
            forward_evidence = ctx.all_evidence[order_idx]
            forward_valid = ctx.all_valid[order_idx]

            if length < order:
                continue

            ends = length - order + 1
            route_offsets = (
                torch.arange(routes, device=symbols.device, dtype=torch.long)
                * alphabet_size**order
            ).view(1, routes)

            for end in range(order - 1, length):
                start = end - order + 1
                for local_position in range(order):
                    source_position = start + local_position
                    place_value = alphabet_size**local_position
                    current_symbol = symbols[:, source_position, :]

                    for bit in range(bit_count):
                        bit_value = 1 << bit
                        # Flip this bit: if currently 1, set to 0; if 0, set to 1
                        is_set = (current_symbol & bit_value) > 0
                        symbol_flipped = torch.where(
                            is_set,
                            current_symbol - bit_value,
                            current_symbol + bit_value,
                        )

                        # Recompute address for this position with flipped symbol
                        base = route_offsets.expand(batch, routes).clone()
                        for lp in range(order):
                            if lp == local_position:
                                base.add_(symbol_flipped * alphabet_size**lp)
                            else:
                                base.add_(
                                    symbols[:, start + lp, :]
                                    * alphabet_size**lp
                                )

                        # Lookup flipped address in address table
                        flipped_sense_rows, flipped_sense_mask, flipped_evidence = (
                            address_table.lookup(
                                order, base, addr_valid[:, end, :]
                            )
                        )

                        # Gather from frozen bank with flipped addresses
                        valid_rows = flipped_sense_rows >= 0
                        safe_rows = flipped_sense_rows.clamp(
                            min=0, max=definition_bank.shape[0] - 1
                        )
                        flipped_gathered = definition_bank.index_select(
                            0, safe_rows.reshape(-1)
                        ).view(
                            batch, routes, address_table.senses_per_address,
                            definition_bank.shape[1],
                        )
                        flipped_valid = (
                            flipped_sense_mask
                            & valid_rows
                            & addr_valid[:, end, :].unsqueeze(-1)
                        )
                        flipped_gathered = (
                            flipped_gathered * flipped_valid.unsqueeze(-1).float()
                        )
                        flipped_anchors = flipped_gathered.view(
                            batch, routes * address_table.senses_per_address,
                            definition_bank.shape[1],
                        )
                        flipped_ev = flipped_evidence.view(
                            batch, routes * address_table.senses_per_address
                        )
                        flipped_vmask = flipped_valid.view(
                            batch, routes * address_table.senses_per_address
                        )

                        # Compute HarMax movement with flipped gather at this
                        # single position, then score the difference
                        pos_hidden = joint_hidden[:, end:end+1, :]  # [B, 1, D]
                        pos_forward_anchors = forward_anchors[:, end:end+1, :, :]
                        pos_forward_ev = forward_evidence[:, end:end+1, :]
                        pos_forward_valid = forward_valid[:, end:end+1, :]

                        # Forward movement (recompute for this position)
                        fwd_move = _signed_harmax_movement(
                            pos_hidden,
                            [pos_forward_anchors],
                            [pos_forward_ev],
                            [pos_forward_valid],
                            harmonic_exponent,
                            epsilon,
                        )  # [B, 1, D]

                        # Flipped movement
                        flip_move = _signed_harmax_movement(
                            pos_hidden,
                            [flipped_anchors.unsqueeze(1)],
                            [flipped_ev.unsqueeze(1)],
                            [flipped_vmask.unsqueeze(1)],
                            harmonic_exponent,
                            epsilon,
                        )  # [B, 1, D]

                        # Score: how much does flipping this bit change the
                        # movement, weighted by upstream gradient
                        score = (
                            grad_movement[:, end:end+1, :]
                            * (flip_move - fwd_move)
                        ).sum(dim=-1)  # [B, 1]

                        probability = torch.sigmoid(
                            surrogate_temperature
                            * logits[:, source_position, :, bit]
                        )  # [B, routes]
                        # Broadcast score from [B, 1] to [B, routes]
                        score_broadcast = score.expand(-1, routes)
                        grad_logits[:, source_position, :, bit].add_(
                            surrogate_scale
                            * surrogate_temperature
                            * probability
                            * (1.0 - probability)
                            * score_broadcast
                        )

        # No gradient for definition_bank (frozen), joint_transfer (handled
        # separately via ordinary autograd), address_table (frozen buffers)
        return (
            grad_logits,  # logits
            None,  # joint_hidden (handled by ordinary autograd through HarMax)
            None,  # address_table
            None,  # definition_bank
            None,  # joint_transfer
            None, None, None, None, None, None,  # scalar args
        )


def _signed_harmax_movement(
    query: Tensor,
    anchor_list: list[Tensor],
    evidence_list: list[Tensor],
    valid_list: list[Tensor],
    harmonic_exponent: float,
    epsilon: float,
) -> Tensor:
    """Compute signed HarMax (y-p) movement from multiple anchor pools.

    For each pool:
        displacement = anchors - query.unsqueeze(-2)
        squared_distance = mean(displacement^2) + epsilon^2
        distance_mass = inverse_distance / sum(inverse_distance)
        target_mass = evidence / sum(evidence)  (y-mass)
        signed_coeff = target_mass - distance_mass  (y - p)
        movement += h * sum(signed_coeff * displacement / squared_distance)

    Total movement is summed across all pools (orders).
    """
    movement = torch.zeros_like(query)
    for anchors, evidence, valid in zip(
        anchor_list, evidence_list, valid_list, strict=True
    ):
        # anchors: [B, T, Q, D], query: [B, T, D]
        displacement = anchors - query.unsqueeze(-2)
        squared_distance = (
            displacement.square().mean(dim=-1) + epsilon**2
        )  # [B, T, Q]

        inverse_distance = squared_distance.pow(-0.5 * harmonic_exponent)
        inverse_distance = inverse_distance * valid.float()
        dist_sum = inverse_distance.sum(dim=-1, keepdim=True) + epsilon
        distance_mass = inverse_distance / dist_sum  # p

        # Evidence (y-mass): clamp to positive, mask by valid
        positive_evidence = evidence.clamp_min(0.0) * valid.float()
        ev_sum = positive_evidence.sum(dim=-1, keepdim=True) + epsilon
        target_mass = positive_evidence / ev_sum  # y

        # Signed coefficient: y - p
        signed_coeff = target_mass - distance_mass  # [B, T, Q]

        # Movement: h * sum(signed_coeff * displacement / squared_distance)
        movement = movement + harmonic_exponent * (
            signed_coeff.unsqueeze(-1)
            * displacement
            / squared_distance.unsqueeze(-1)
        ).sum(dim=-2)

    return movement


class _RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        scale = values.square().mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(scale + self.epsilon) * self.weight


@dataclass(frozen=True)
class DefinitionLnGramStats:
    symbols: Tensor
    addresses: dict[int, Tensor]
    valid: dict[int, Tensor]
    movement: Tensor


class DefinitionLnGram(nn.Module):
    """LNGram definition memory with frozen bank gather and signed HarMax.

    Routes in joint space (via JTD live_to_joint), looks up sense rows in
    AddressRecordTable, gathers frozen definition vectors, applies signed
    HarMax (y-p) contraction, and outputs via JTD joint_to_live.

    J_h^T (training gradient through W_q) is distinct from joint_to_live
    (runtime movement map) — no tied-map constraint.
    """

    def __init__(
        self,
        model_width: int,
        memory_width: int,
        *,
        bits_per_route: int = 4,
        orders: Iterable[int] = (2, 3),
        senses_per_address: int = 4,
        harmonic_exponent: float = 2.0,
        surrogate_temperature: float = 1.0,
        surrogate_scale: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if model_width % bits_per_route:
            raise ValueError("model_width must be divisible by bits_per_route")
        self.model_width = int(model_width)
        self.memory_width = int(memory_width)
        self.bits_per_route = int(bits_per_route)
        self.route_count = model_width // bits_per_route
        self.alphabet_size = 2**bits_per_route
        self.orders = tuple(sorted(set(int(o) for o in orders)))
        if not self.orders or min(self.orders) < 1:
            raise ValueError("orders must contain positive integers")
        self.senses_per_address = int(senses_per_address)
        self.harmonic_exponent = float(harmonic_exponent)
        self.surrogate_temperature = float(surrogate_temperature)
        self.surrogate_scale = float(surrogate_scale)
        self.epsilon = float(epsilon)

        # JTD: live_to_joint for routing, joint_to_live for output
        # These are independent maps — no tied constraint
        self.joint_transfer = JointTransferDomain(
            model_width,
            memory_width=memory_width,
        )

        # Address record table (frozen buffers)
        self.address_table = AddressRecordTable(
            self.route_count,
            self.alphabet_size,
            self.orders,
            self.senses_per_address,
        )

        # Routing projection W_q: operates in joint space
        self.input_norm = _RMSNorm(memory_width)
        self.address_projection = nn.Linear(memory_width, model_width, bias=False)

        # Output gate
        self.output_gate = nn.Parameter(torch.tensor(1e-3))

        # Frozen definition bank: set via attach_definition_bank
        self.definition_bank: Tensor | None = None

    @torch.no_grad()
    def attach_definition_bank(self, bank: Tensor) -> None:
        if bank.ndim != 2 or bank.shape[-1] != self.memory_width:
            raise ValueError(
                f"Definition bank must be [N_senses, {self.memory_width}]"
            )
        self.definition_bank = bank

    def forward(
        self,
        hidden: Tensor,
        *,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, DefinitionLnGramStats]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.model_width:
            raise ValueError(
                f"Expected hidden [B,T,{self.model_width}]"
            )
        if self.definition_bank is None:
            raise RuntimeError(
                "DefinitionLnGram requires attach_definition_bank before forward"
            )

        # Project live state to joint space for routing
        joint_hidden = self.joint_transfer.source_to_joint(hidden, "live")

        # Route in joint space
        normalized = self.input_norm(joint_hidden)
        logits = self.address_projection(normalized).view(
            hidden.shape[0],
            hidden.shape[1],
            self.route_count,
            self.bits_per_route,
        )

        # Counterfactual bank gather + signed HarMax (custom autograd)
        movement_joint = _CounterfactualBankGather.apply(
            logits,
            joint_hidden,
            self.address_table,
            self.definition_bank,
            self.joint_transfer,
            self.orders,
            self.bits_per_route,
            self.harmonic_exponent,
            self.epsilon,
            self.surrogate_temperature,
            self.surrogate_scale,
        )

        # Map joint movement back to live space via joint_to_live
        # This is the RUNTIME movement map, distinct from J_h^T training gradient
        movement_live = self.joint_transfer.movement_to_live(movement_joint)

        # Gated residual update
        output = hidden + torch.tanh(self.output_gate) * movement_live

        if return_stats:
            symbols = _pack_bits(logits > 0)
            address_stats: dict[int, Tensor] = {}
            valid_stats: dict[int, Tensor] = {}
            for order in self.orders:
                addresses, valid = _addresses(symbols, order, self.alphabet_size)
                address_stats[order] = addresses
                valid_stats[order] = valid
            return output, DefinitionLnGramStats(
                symbols=symbols,
                addresses=address_stats,
                valid=valid_stats,
                movement=movement_joint,
            )
        return output
