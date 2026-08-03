"""Universal directions, shared skills, and high-dimensional experts.

All three objects live in weight/operation space.  They never consume token IDs
as semantic labels.  The current 2,048D activation selects operations, while
the two physical recurrent blocks carry the changing thought.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.working_adapter requires PyTorch. Install torch>=2.7."
    ) from error


@dataclass(frozen=True)
class DirectionRouting:
    indices: Tensor
    weights: Tensor


@dataclass(frozen=True)
class SkillExpertStats:
    universal: DirectionRouting
    skills: DirectionRouting
    experts: DirectionRouting
    universal_gate: Tensor
    skill_gate: Tensor
    expert_gate: Tensor


def _signed_topk(
    scores: Tensor,
    *,
    top_k: int,
    valid: Tensor | None = None,
    epsilon: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    if valid is None:
        valid = torch.ones_like(scores, dtype=torch.bool)
    magnitude = scores.abs().masked_fill(~valid, -torch.inf)
    count = min(int(top_k), scores.shape[-1])
    indices = magnitude.topk(count, dim=-1).indices
    selected_scores = scores.gather(-1, indices)
    selected_valid = valid.gather(-1, indices)
    selected_scores = selected_scores * selected_valid
    denominator = selected_scores.abs().sum(dim=-1, keepdim=True)
    selected_weights = selected_scores / (denominator + epsilon)
    dense_weights = torch.zeros_like(scores).scatter(-1, indices, selected_weights)
    return indices, selected_weights, dense_weights


class UniversalDirectionBank(nn.Module):
    """Frozen low-rank weight directions for both physical blocks."""

    def __init__(
        self,
        width: int,
        *,
        rank: int,
        top_k: int,
        block_count: int = 2,
        bootstrap_seed: int | None = None,
        residual_initialization_gain: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.rank = int(rank)
        self.top_k = int(top_k)
        self.block_count = int(block_count)
        self.residual_initialization_gain = float(residual_initialization_gain)
        if self.residual_initialization_gain <= 0:
            raise ValueError("residual_initialization_gain must be positive")
        self.epsilon = float(epsilon)
        alpha = torch.zeros(block_count, rank, width)
        beta = torch.zeros(block_count, rank, width)
        loaded = False
        if rank and bootstrap_seed is not None:
            generator = torch.Generator().manual_seed(int(bootstrap_seed))
            for block in range(block_count):
                a, _ = torch.linalg.qr(
                    torch.randn(width, rank, generator=generator),
                    mode="reduced",
                )
                b, _ = torch.linalg.qr(
                    torch.randn(width, rank, generator=generator),
                    mode="reduced",
                )
                alpha[block] = a.T
                beta[block] = self.residual_initialization_gain * b.T
            loaded = True
        self.alpha = nn.Parameter(alpha, requires_grad=False)
        self.beta = nn.Parameter(beta, requires_grad=False)
        self.register_buffer(
            "basis_loaded",
            torch.tensor(loaded, dtype=torch.bool),
        )

    @torch.no_grad()
    def load_directions(self, alpha: Tensor, beta: Tensor) -> None:
        expected = (self.block_count, self.rank, self.width)
        if tuple(alpha.shape) != expected or tuple(beta.shape) != expected:
            raise ValueError(
                f"Universal directions must both have shape {expected}"
            )
        self.alpha.copy_(alpha.to(device=self.alpha.device, dtype=self.alpha.dtype))
        self.beta.copy_(
            self.residual_initialization_gain
            * beta.to(device=self.beta.device, dtype=self.beta.dtype)
        )
        self.basis_loaded.fill_(True)

    def forward(
        self,
        hidden: Tensor,
        *,
        block_index: int,
    ) -> tuple[Tensor, DirectionRouting]:
        if self.rank == 0 or not bool(self.basis_loaded):
            empty_indices = torch.empty(
                *hidden.shape[:2], 0, dtype=torch.long, device=hidden.device
            )
            empty_weights = hidden.new_zeros(*hidden.shape[:2], 0)
            return torch.zeros_like(hidden), DirectionRouting(
                empty_indices, empty_weights
            )
        alpha = F.normalize(self.alpha[block_index], dim=-1)
        beta = self.beta[block_index]
        normalized = F.normalize(hidden, dim=-1)
        scores = torch.einsum("btd,rd->btr", normalized, alpha)
        indices, selected_weights, dense_weights = _signed_topk(
            scores,
            top_k=self.top_k,
            epsilon=self.epsilon,
        )
        response = torch.einsum("btd,rd->btr", hidden, self.alpha[block_index])
        update = torch.einsum("btr,btr,rd->btd", dense_weights, response, beta)
        return update, DirectionRouting(indices, selected_weights)


class SharedSkillAdapters(nn.Module):
    """Trainable LoRA skills extending the frozen universal directions."""

    def __init__(
        self,
        width: int,
        *,
        skill_count: int,
        skill_rank: int,
        top_k: int,
        block_count: int = 2,
        residual_initialization_gain: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.skill_count = int(skill_count)
        self.skill_rank = int(skill_rank)
        self.top_k = int(top_k)
        self.block_count = int(block_count)
        self.residual_initialization_gain = float(residual_initialization_gain)
        if self.residual_initialization_gain <= 0:
            raise ValueError("residual_initialization_gain must be positive")
        self.epsilon = float(epsilon)
        self.anchors = nn.Parameter(torch.empty(skill_count, width))
        self.alpha = nn.Parameter(
            torch.empty(block_count, skill_count, skill_rank, width)
        )
        self.beta = nn.Parameter(
            torch.empty(block_count, skill_count, width, skill_rank)
        )
        nn.init.normal_(self.anchors, std=width**-0.5)
        nn.init.normal_(self.alpha, std=width**-0.5)
        nn.init.normal_(self.beta, std=width**-0.5)
        with torch.no_grad():
            self.alpha.mul_(self.residual_initialization_gain)
            self.beta.mul_(self.residual_initialization_gain)

    def forward(
        self,
        hidden: Tensor,
        *,
        block_index: int,
    ) -> tuple[Tensor, DirectionRouting, Tensor]:
        scores = torch.einsum(
            "btd,sd->bts",
            F.normalize(hidden, dim=-1),
            F.normalize(self.anchors, dim=-1),
        )
        indices, selected_weights, dense_weights = _signed_topk(
            scores,
            top_k=self.top_k,
            epsilon=self.epsilon,
        )
        update = torch.zeros_like(hidden)
        for skill_id in torch.unique(indices).tolist():
            skill_weight = dense_weights[..., skill_id]
            low_rank = F.linear(hidden, self.alpha[block_index, skill_id])
            skill_update = F.linear(low_rank, self.beta[block_index, skill_id])
            update = update + skill_weight.unsqueeze(-1) * skill_update
        return update, DirectionRouting(indices, selected_weights), dense_weights


class DendriticBranch(nn.Module):
    def __init__(
        self,
        width: int,
        hidden_width: int,
        *,
        residual_initialization_gain: float = 1.0,
    ) -> None:
        super().__init__()
        self.residual_initialization_gain = float(residual_initialization_gain)
        if self.residual_initialization_gain <= 0:
            raise ValueError("residual_initialization_gain must be positive")
        self.content = nn.Linear(width, hidden_width)
        self.gate = nn.Linear(width, hidden_width)
        self.output = nn.Linear(hidden_width, width, bias=False)
        self.evidence = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.evidence, std=width**-0.5)
        nn.init.normal_(self.output.weight, std=hidden_width**-0.5)
        with torch.no_grad():
            self.content.weight.mul_(self.residual_initialization_gain)
            self.gate.weight.mul_(self.residual_initialization_gain)
            self.output.weight.mul_(self.residual_initialization_gain)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        branch = torch.tanh(self.content(values)) * torch.sigmoid(self.gate(values))
        output = self.output(branch)
        evidence = (
            F.normalize(output, dim=-1)
            * F.normalize(self.evidence, dim=-1)
        ).sum(dim=-1)
        return output, evidence


class DendriticExpert(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        hidden_width: int,
        branch_count: int,
        block_count: int = 2,
        residual_initialization_gain: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DendriticBranch(
                            width,
                            hidden_width,
                            residual_initialization_gain=residual_initialization_gain,
                        )
                        for _ in range(branch_count)
                    ]
                )
                for _ in range(block_count)
            ]
        )

    def forward(self, values: Tensor, *, block_index: int) -> Tensor:
        outputs = []
        scores = []
        for branch in self.blocks[block_index]:
            output, evidence = branch(values)
            outputs.append(output)
            scores.append(evidence)
        output_stack = torch.stack(outputs, dim=-2)
        score_stack = torch.stack(scores, dim=-1)
        weights = score_stack / (
            score_stack.abs().sum(dim=-1, keepdim=True) + self.epsilon
        )
        return (weights.unsqueeze(-1) * output_stack).sum(dim=-2)


class ConditionalExpertBank(nn.Module):
    """High-dimensional experts reached through skill adjacency."""

    def __init__(
        self,
        width: int,
        *,
        skill_count: int,
        expert_count: int,
        hidden_width: int,
        branch_count: int,
        top_k: int,
        block_count: int = 2,
        residual_initialization_gain: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.skill_count = int(skill_count)
        self.expert_count = int(expert_count)
        self.top_k = int(top_k)
        self.epsilon = float(epsilon)
        self.anchors = nn.Parameter(torch.empty(expert_count, width))
        nn.init.normal_(self.anchors, std=width**-0.5)
        self.experts = nn.ModuleList(
            [
                DendriticExpert(
                    width,
                    hidden_width=hidden_width,
                    branch_count=branch_count,
                    block_count=block_count,
                    residual_initialization_gain=residual_initialization_gain,
                    epsilon=epsilon,
                )
                for _ in range(expert_count)
            ]
        )
        adjacency = torch.zeros(skill_count, expert_count, dtype=torch.bool)
        for skill_id in range(skill_count):
            adjacency[skill_id, skill_id % expert_count] = True
        for expert_id in range(expert_count):
            adjacency[expert_id % skill_count, expert_id] = True
        self.register_buffer("skill_adjacency", adjacency)

    @torch.no_grad()
    def set_adjacency(self, adjacency: Tensor) -> None:
        expected = (self.skill_count, self.expert_count)
        if tuple(adjacency.shape) != expected:
            raise ValueError(f"Expert adjacency must have shape {expected}")
        value = adjacency.to(
            device=self.skill_adjacency.device,
            dtype=torch.bool,
        )
        if not value.any(dim=-1).all():
            raise ValueError("Every skill requires at least one adjacent expert")
        self.skill_adjacency.copy_(value)

    def forward(
        self,
        hidden: Tensor,
        *,
        block_index: int,
        active_skill_indices: Tensor,
    ) -> tuple[Tensor, DirectionRouting]:
        candidates = self.skill_adjacency[active_skill_indices].any(dim=-2)
        scores = torch.einsum(
            "btd,ed->bte",
            F.normalize(hidden, dim=-1),
            F.normalize(self.anchors, dim=-1),
        )
        indices, selected_weights, dense_weights = _signed_topk(
            scores,
            top_k=self.top_k,
            valid=candidates,
            epsilon=self.epsilon,
        )

        flat_hidden = hidden.reshape(-1, self.width)
        flat_weights = dense_weights.reshape(-1, self.expert_count)
        flat_update = torch.zeros_like(flat_hidden)
        for expert_id in torch.unique(indices).tolist():
            weights = flat_weights[:, expert_id]
            selected = weights != 0
            if not bool(selected.any()):
                continue
            positions = selected.nonzero(as_tuple=False).flatten()
            values = flat_hidden.index_select(0, positions)
            expert_update = self.experts[expert_id](
                values,
                block_index=block_index,
            )
            weighted = expert_update * weights.index_select(0, positions).unsqueeze(-1)
            flat_update = flat_update.index_add(0, positions, weighted)
        return flat_update.view_as(hidden), DirectionRouting(
            indices, selected_weights
        )


class SkillExpertSystem(nn.Module):
    """Full universal -> skill -> expert operation hierarchy."""

    def __init__(
        self,
        width: int,
        *,
        universal_rank: int,
        universal_top_k: int,
        universal_bootstrap_seed: int | None,
        skill_count: int,
        skill_rank: int,
        skill_top_k: int,
        expert_count: int,
        expert_hidden_width: int,
        expert_branches: int,
        expert_top_k: int,
        residual_initialization_gain: float = 1.0,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.universal = UniversalDirectionBank(
            width,
            rank=universal_rank,
            top_k=universal_top_k,
            bootstrap_seed=universal_bootstrap_seed,
            residual_initialization_gain=residual_initialization_gain,
            epsilon=epsilon,
        )
        self.skills = SharedSkillAdapters(
            width,
            skill_count=skill_count,
            skill_rank=skill_rank,
            top_k=skill_top_k,
            residual_initialization_gain=residual_initialization_gain,
            epsilon=epsilon,
        )
        self.experts = ConditionalExpertBank(
            width,
            skill_count=skill_count,
            expert_count=expert_count,
            hidden_width=expert_hidden_width,
            branch_count=expert_branches,
            top_k=expert_top_k,
            residual_initialization_gain=residual_initialization_gain,
            epsilon=epsilon,
        )
        self.universal_gate = nn.Parameter(torch.tensor(1e-3))
        self.skill_gate = nn.Parameter(torch.tensor(1e-3))
        self.expert_gate = nn.Parameter(torch.tensor(1e-3))

    def forward(
        self,
        hidden: Tensor,
        *,
        block_index: int,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SkillExpertStats]:
        universal, universal_route = self.universal(
            hidden, block_index=block_index
        )
        skill, skill_route, _ = self.skills(hidden, block_index=block_index)
        expert, expert_route = self.experts(
            hidden,
            block_index=block_index,
            active_skill_indices=skill_route.indices,
        )
        update = (
            torch.tanh(self.universal_gate) * universal
            + torch.tanh(self.skill_gate) * skill
            + torch.tanh(self.expert_gate) * expert
        )
        if return_stats:
            return update, SkillExpertStats(
                universal=universal_route,
                skills=skill_route,
                experts=expert_route,
                universal_gate=torch.tanh(self.universal_gate),
                skill_gate=torch.tanh(self.skill_gate),
                expert_gate=torch.tanh(self.expert_gate),
            )
        return update
