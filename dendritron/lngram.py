"""PyTorch LNGram memory with exact hard lookup and surrogate gradients.

The forward path follows the attached LNGram paper:

    hidden state -> RMSNorm -> trainable projection -> hard route symbols
    -> exact route-partitioned 2/3-gram lookup -> context-aware readout

The custom autograd function implements the paper's one-bit counterfactual
surrogate for the routing logits. Selected table rows and readout parameters
still receive ordinary exact gradients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

try:
    import torch
    from torch import Tensor, nn
    from torch.autograd import Function
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.lngram requires PyTorch. Install torch>=2.7."
    ) from error


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


class _CounterfactualLookup(Function):
    @staticmethod
    def forward(
        ctx: Any,
        logits: Tensor,
        table_weight: Tensor,
        order: int,
        bits_per_route: int,
        temperature: float,
        surrogate_scale: float,
    ) -> Tensor:
        alphabet_size = 2**bits_per_route
        symbols = _pack_bits(logits > 0)
        addresses, valid = _addresses(symbols, order, alphabet_size)
        selected = table_weight.index_select(0, addresses.reshape(-1)).view(
            *addresses.shape,
            table_weight.shape[1],
        )
        selected = selected * valid.unsqueeze(-1)
        ctx.save_for_backward(logits, symbols, addresses, valid, table_weight)
        ctx.order = int(order)
        ctx.bits_per_route = int(bits_per_route)
        ctx.temperature = float(temperature)
        ctx.surrogate_scale = float(surrogate_scale)
        return selected

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor):
        logits, symbols, addresses, valid, table_weight = ctx.saved_tensors
        order = ctx.order
        bit_count = ctx.bits_per_route
        alphabet_size = 2**bit_count
        batch, length, routes, memory_width = grad_output.shape

        grad_table = torch.zeros_like(table_weight)
        flat_valid = valid.reshape(-1)
        grad_table.index_add_(
            0,
            addresses.reshape(-1)[flat_valid],
            grad_output.reshape(-1, memory_width)[flat_valid],
        )

        grad_logits = torch.zeros_like(logits)
        if length >= order and ctx.needs_input_grad[0]:
            route_offsets = (
                torch.arange(routes, device=symbols.device, dtype=torch.long)
                * alphabet_size**order
            ).view(1, routes)
            for end in range(order - 1, length):
                start = end - order + 1
                base = route_offsets.expand(batch, routes).clone()
                for local_position in range(order):
                    base.add_(
                        symbols[:, start + local_position, :]
                        * alphabet_size**local_position
                    )
                upstream = grad_output[:, end, :, :]

                for local_position in range(order):
                    source_position = start + local_position
                    place_value = alphabet_size**local_position
                    current_symbol = symbols[:, source_position, :]
                    for bit in range(bit_count):
                        bit_value = 1 << bit
                        symbol_zero = torch.bitwise_and(current_symbol, ~bit_value)
                        symbol_one = torch.bitwise_or(current_symbol, bit_value)
                        address_zero = base + (symbol_zero - current_symbol) * place_value
                        address_one = base + (symbol_one - current_symbol) * place_value
                        value_zero = table_weight.index_select(
                            0, address_zero.reshape(-1)
                        ).view(batch, routes, memory_width)
                        value_one = table_weight.index_select(
                            0, address_one.reshape(-1)
                        ).view(batch, routes, memory_width)
                        score = (
                            upstream * (value_one - value_zero)
                        ).sum(dim=-1)
                        probability = torch.sigmoid(
                            ctx.temperature
                            * logits[:, source_position, :, bit]
                        )
                        grad_logits[:, source_position, :, bit].add_(
                            ctx.surrogate_scale
                            * ctx.temperature
                            * probability
                            * (1.0 - probability)
                            * score
                        )

        return grad_logits, grad_table, None, None, None, None


@dataclass(frozen=True)
class LNGramStats:
    symbols: Tensor
    addresses: dict[int, Tensor]
    valid: dict[int, Tensor]
    gates: dict[int, Tensor]


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        scale = values.square().mean(dim=-1, keepdim=True)
        return values * torch.rsqrt(scale + self.epsilon) * self.weight


class LNGramMemory(nn.Module):
    """Single-table, multi-route LNGram residual memory.

    ``readout_mode="distance"`` is Dendritron's softmax-free geometric
    adaptation. ``readout_mode="paper_sigmoid"`` preserves the paper's
    single-table readout for a controlled ablation.
    """

    def __init__(
        self,
        model_width: int,
        *,
        bits_per_route: int = 4,
        orders: Iterable[int] = (2, 3),
        route_memory_width: int | None = None,
        surrogate_temperature: float = 1.0,
        surrogate_scale: float = 1.0,
        convolution_kernel: int = 4,
        readout_mode: str = "distance",
        readout_epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if model_width % bits_per_route:
            raise ValueError("model_width must be divisible by bits_per_route")
        self.model_width = int(model_width)
        self.bits_per_route = int(bits_per_route)
        self.route_count = model_width // bits_per_route
        self.alphabet_size = 2**bits_per_route
        self.orders = tuple(sorted(set(int(value) for value in orders)))
        if not self.orders or min(self.orders) < 1:
            raise ValueError("orders must contain positive integers")
        self.route_memory_width = (
            int(route_memory_width)
            if route_memory_width is not None
            else bits_per_route
        )
        self.retrieval_width = self.route_count * self.route_memory_width
        self.surrogate_temperature = float(surrogate_temperature)
        self.surrogate_scale = float(surrogate_scale)
        if readout_mode not in {"distance", "paper_sigmoid"}:
            raise ValueError("readout_mode must be distance or paper_sigmoid")
        self.readout_mode = readout_mode
        self.readout_epsilon = float(readout_epsilon)

        self.input_norm = RMSNorm(model_width)
        self.address_projection = nn.Linear(model_width, model_width, bias=False)
        self.tables = nn.ParameterDict()
        for order in self.orders:
            rows = self.route_count * self.alphabet_size**order
            weight = nn.Parameter(
                torch.empty(rows, self.route_memory_width),
            )
            nn.init.normal_(weight, mean=0.0, std=1.0 / math.sqrt(model_width))
            self.tables[str(order)] = weight

        self.key_projection = nn.Linear(self.retrieval_width, model_width)
        self.value_projection = nn.Linear(self.retrieval_width, model_width)
        self.query_norm = RMSNorm(model_width)
        self.key_norm = RMSNorm(model_width)
        self.output_norm = RMSNorm(model_width)

        padding = convolution_kernel - 1
        self.causal_padding = padding
        self.depthwise_conv = nn.Conv1d(
            model_width,
            model_width,
            kernel_size=convolution_kernel,
            groups=model_width,
            padding=padding,
        )
        self.output_gate = nn.Parameter(torch.tensor(1e-3))

    @property
    def table_row_counts(self) -> dict[int, int]:
        return {
            order: self.route_count * self.alphabet_size**order
            for order in self.orders
        }

    def forward(
        self,
        hidden: Tensor,
        *,
        forced_addresses: Mapping[int, Tensor] | None = None,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, LNGramStats]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.model_width:
            raise ValueError(
                f"Expected hidden [B, T, {self.model_width}], found {tuple(hidden.shape)}"
            )

        normalized = self.input_norm(hidden)
        logits = self.address_projection(normalized).view(
            hidden.shape[0],
            hidden.shape[1],
            self.route_count,
            self.bits_per_route,
        )
        symbols = _pack_bits(logits > 0)
        query = self.query_norm(hidden)

        values_by_order: list[Tensor] = []
        scores_by_order: list[Tensor] = []
        address_stats: dict[int, Tensor] = {}
        valid_stats: dict[int, Tensor] = {}
        gate_stats: dict[int, Tensor] = {}
        for order in self.orders:
            retrieval = _CounterfactualLookup.apply(
                logits,
                self.tables[str(order)],
                order,
                self.bits_per_route,
                self.surrogate_temperature,
                self.surrogate_scale,
            )
            forced = None if forced_addresses is None else forced_addresses.get(order)
            if forced is not None:
                if forced.shape != symbols.shape:
                    raise ValueError(
                        f"Forced LNGram addresses for order {order} must be "
                        f"{tuple(symbols.shape)}"
                    )
                forced = forced.to(device=hidden.device, dtype=torch.long)
                forced_valid = forced >= 0
                if bool(
                    (forced[forced_valid] >= self.tables[str(order)].shape[0]).any()
                ):
                    raise ValueError(
                        f"Forced LNGram address exceeds the order-{order} table"
                    )
                safe_forced = forced.clamp(
                    min=0,
                    max=self.tables[str(order)].shape[0] - 1,
                )
                forced_values = self.tables[str(order)].index_select(
                    0, safe_forced.reshape(-1)
                ).view(*safe_forced.shape, self.route_memory_width)
                retrieval = torch.where(
                    forced_valid.unsqueeze(-1),
                    forced_values,
                    retrieval,
                )
            flattened = retrieval.flatten(start_dim=-2)
            key = self.key_norm(self.key_projection(flattened))
            value = self.value_projection(flattened)
            squared_distance = (query - key).square().mean(dim=-1)
            score = (squared_distance + self.readout_epsilon).reciprocal()
            valid_position = (
                torch.arange(hidden.shape[1], device=hidden.device) >= order - 1
            ).view(1, -1)
            score = score * valid_position
            values_by_order.append(value)
            scores_by_order.append(score)

            if return_stats:
                addresses, valid = _addresses(symbols, order, self.alphabet_size)
                if forced is not None:
                    forced_valid = forced >= 0
                    addresses = torch.where(forced_valid, forced, addresses)
                    valid = valid | forced_valid
                address_stats[order] = addresses
                valid_stats[order] = valid

        score_stack = torch.stack(scores_by_order, dim=-1)
        if self.readout_mode == "paper_sigmoid":
            weights = torch.sigmoid(score_stack)
            valid_orders = torch.stack(
                [
                    (
                        torch.arange(hidden.shape[1], device=hidden.device)
                        >= order - 1
                    ).view(1, -1)
                    for order in self.orders
                ],
                dim=-1,
            )
            weights = weights * valid_orders
        else:
            valid_orders = torch.stack(
                [
                    (
                        torch.arange(hidden.shape[1], device=hidden.device)
                        >= order - 1
                    ).view(1, -1)
                    for order in self.orders
                ],
                dim=-1,
            )
            score_stack = score_stack * valid_orders
            denominator = score_stack.sum(dim=-1, keepdim=True)
            weights = score_stack / (denominator + self.readout_epsilon)

        fused = torch.zeros_like(hidden)
        for index, (order, value) in enumerate(
            zip(self.orders, values_by_order, strict=True)
        ):
            fused = fused + weights[..., index].unsqueeze(-1) * value
            if return_stats:
                gate_stats[order] = weights[..., index]

        sequence = self.output_norm(fused).transpose(1, 2)
        convolution = self.depthwise_conv(sequence)
        if self.causal_padding:
            convolution = convolution[:, :, : hidden.shape[1]]
        memory_update = fused + torch.nn.functional.silu(convolution.transpose(1, 2))
        output = hidden + torch.tanh(self.output_gate) * memory_update

        if return_stats:
            return output, LNGramStats(
                symbols=symbols,
                addresses=address_stats,
                valid=valid_stats,
                gates=gate_stats,
            )
        return output
