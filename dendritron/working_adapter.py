"""Shared/private working-LoRA skill layer and high-dimensional experts.

Implements the shared-plus-private skill decomposition:

    L_j = b_j a_j^T                    (frozen shared matched operators)
    ΔW_s^shared = Σ_j q_{s,j} L_j      (per-skill coefficients over shared basis)
    X_s = B_s^priv A_s^priv            (per-skill private LoRA residual)
    ΔW_s = ΔW_s^shared + X_s           (total skill weight update)
    Δh_t = Σ_s g_{t,s} ΔW_s h_t        (runtime routing)

The shared basis {L_j} is frozen during tasks.  Per-skill coefficients
q_{s,j} and private LoRAs (B_s^priv, A_s^priv) are offline-learnable
and frozen during inference.  Only routing g_{t,s} is dynamic at runtime.

Cross-pair terms are structurally zero: the shared basis is common to
all skills (no factor mixing), and each private LoRA is independent
per skill.

The production forward path uses gather-then-compute: only the top-k
active skills' coefficient and private factor blocks are gathered,
scaling CPU cost with top_k rather than max_skill_slots.

Experts and branches consume the same selected skill state.
All objects live in weight/operation space.
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

from .branch_operators import (
    BranchOutcome,
    DeductiveBranchInstance,
    DeductiveBranchOperator,
    combine_branch_movements,
)


@dataclass(frozen=True)
class DirectionRouting:
    indices: Tensor
    weights: Tensor


@dataclass(frozen=True)
class RoutingContext:
    """Single routing decision for one block visit.

    Produced once per block visit and reused by branch contraction,
    composed working LoRA, and expert soma.
    """
    skill_indices: Tensor          # [B, T, skill_top_k]
    skill_weights: Tensor          # [B, T, skill_top_k]
    skill_dense_weights: Tensor    # [B, T, max_skill_slots]
    skill_mask: Tensor             # [B, T, max_skill_slots] bool


@dataclass(frozen=True)
class BranchContractionStats:
    """Stats from the branch contraction phase (first sublayer)."""
    movement: Tensor
    branch_count: int
    branch_evidences: tuple[Tensor, ...]
    branch_residuals: tuple[Tensor, ...]


@dataclass(frozen=True)
class SkillExpertStats:
    skills: DirectionRouting
    experts: DirectionRouting
    skill_gate: Tensor
    expert_gate: Tensor
    branch_contraction: BranchContractionStats | None = None


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


class ComposedWorkingLoRA(nn.Module):
    """Shared/private working-LoRA skill layer with slot population mask.

    The effective weight update for skill slot s in block b is:

        ΔW_{s,b} = Σ_j q_{s,j} b_{b,j} a_{b,j}^T + B_{s,b}^priv A_{s,b}^priv

    where:
        a_{b,j}, b_{b,j}:  frozen shared matched read/write directions (block-specific)
        q_{s,j}:           per-skill coefficient over shared basis (shared across blocks)
        A_{s,b}^priv:      private input factor [R_max, D] (padded, per-skill rank masked)
        B_{s,b}^priv:      private output factor [D, R_max] (padded, per-skill rank masked)

    Runtime routing g_{t,s} selects and weights populated skill slots per token:

        Δh = Σ_s g_{t,s} ΔW_{s,b} h

    Only populated slots participate in routing.  Each slot's private LoRA
    is padded to max_private_lora_rank for efficient batching; a per-skill
    rank mask zeroes the padded dimensions.

    The shared basis is computed once per token and reused across all
    active skills.  Private LoRA factors are gathered per active skill.
    Cross-pair leakage is structurally impossible: the shared basis is
    common (not factor-mixed) and private LoRAs are per-skill independent.
    """

    def __init__(
        self,
        width: int,
        *,
        max_skill_slots: int,
        shared_basis_count: int,
        max_private_lora_rank: int,
        top_k: int,
        block_count: int = 2,
        init_mode: str = "smoke",
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.max_skill_slots = int(max_skill_slots)
        self.shared_basis_count = int(shared_basis_count)
        self.max_private_lora_rank = int(max_private_lora_rank)
        self.top_k = int(top_k)
        self.block_count = int(block_count)
        self.init_mode = str(init_mode)
        self.epsilon = float(epsilon)

        # Frozen shared matched operators: L_j = b_j a_j^T
        # shared_alpha: [block_count, K, D]  (a_j, input-side, block-specific)
        # shared_beta:  [block_count, D, K]  (b_j, output-side, block-specific)
        # These come from offline SVD of stacked successful reasoning transitions.
        self.shared_alpha = nn.Parameter(
            torch.zeros(block_count, shared_basis_count, width),
            requires_grad=False,
        )
        self.shared_beta = nn.Parameter(
            torch.zeros(block_count, width, shared_basis_count),
            requires_grad=False,
        )

        # Per-skill coefficients over shared basis: [S, K] (frozen)
        # q_{s,j}: how skill slot s uses shared operator j.  Shared across blocks
        # to preserve skill identity; block-specific structure lives in a_{b,j}, b_{b,j}.
        self.skill_coeffs = nn.Parameter(
            torch.zeros(max_skill_slots, shared_basis_count),
            requires_grad=False,
        )

        # Per-skill private LoRA residual: X_s = B_s^priv A_s^priv
        # private_alpha: [block_count, S, R_max, D]  (A_s^priv, input-side)
        # private_beta:  [block_count, S, D, R_max]  (B_s^priv, output-side)
        # Padded to max_private_lora_rank for efficient batching.
        # Per-skill actual rank tracked in private_ranks buffer; padded dims masked.
        # Offline-learnable, frozen during inference.
        self.private_alpha = nn.Parameter(
            torch.zeros(block_count, max_skill_slots, max_private_lora_rank, width),
            requires_grad=False,
        )
        self.private_beta = nn.Parameter(
            torch.zeros(block_count, max_skill_slots, width, max_private_lora_rank),
            requires_grad=False,
        )

        # Skill anchors for geometric routing: [max_skill_slots, width]
        # Trainable during offline fitting; frozen during production inference.
        self.anchors = nn.Parameter(torch.empty(max_skill_slots, width))

        # Slot population mask: [max_skill_slots] bool
        # True for slots that have been populated with a learned skill.
        # Only populated slots participate in routing.
        # In smoke mode, all slots are populated for structural testing.
        self.register_buffer(
            "slot_populated",
            torch.ones(max_skill_slots, dtype=torch.bool),
        )

        # Per-skill private rank: [max_skill_slots] int
        # Actual rank used by each skill (<= max_private_lora_rank).
        # Padded dimensions beyond this rank are masked to zero.
        # In smoke mode, all slots use max_private_lora_rank.
        self.register_buffer(
            "private_ranks",
            torch.full((max_skill_slots,), max_private_lora_rank, dtype=torch.long),
        )

        # Load-state buffer
        self.register_buffer(
            "factors_loaded",
            torch.tensor(False, dtype=torch.bool),
        )

        if init_mode == "smoke":
            self._smoke_init()
        elif init_mode == "production":
            # Production starts empty; must load factors + anchors before forward
            nn.init.normal_(self.anchors, std=width**-0.5)
            # No slots populated until load_skill_factors is called
            self.slot_populated.fill_(False)
            self.private_ranks.fill_(0)
        else:
            raise ValueError(f"init_mode must be 'smoke' or 'production', got {init_mode!r}")

    def _smoke_init(self) -> None:
        """Random initialization for structural smoke tests only.

        All slots are populated with full-rank private LoRAs for testing.
        """
        generator = torch.Generator().manual_seed(42)
        for block in range(self.block_count):
            # Shared basis: K QR-orthonormalized matched operator directions
            a_shared, _ = torch.linalg.qr(
                torch.randn(self.width, self.shared_basis_count, generator=generator),
                mode="reduced",
            )
            b_shared, _ = torch.linalg.qr(
                torch.randn(self.width, self.shared_basis_count, generator=generator),
                mode="reduced",
            )
            # shared_alpha[block] = Q^T → [K, D]
            self.shared_alpha.data[block] = a_shared.T
            # shared_beta[block] = Q → [D, K]
            self.shared_beta.data[block] = b_shared

            # Private LoRA: per-skill QR-orthonormalized factors
            for slot in range(self.max_skill_slots):
                a_priv, _ = torch.linalg.qr(
                    torch.randn(self.width, self.max_private_lora_rank, generator=generator),
                    mode="reduced",
                )
                b_priv, _ = torch.linalg.qr(
                    torch.randn(self.width, self.max_private_lora_rank, generator=generator),
                    mode="reduced",
                )
                self.private_alpha.data[block, slot] = a_priv.T
                self.private_beta.data[block, slot] = b_priv

        # Per-skill coefficients: random normal [S, K] — shared across blocks
        self.skill_coeffs.data = torch.randn(
            self.max_skill_slots, self.shared_basis_count, generator=generator
        ) * (self.shared_basis_count ** -0.5)

        nn.init.normal_(self.anchors, std=self.width**-0.5)
        # All slots populated, all at max rank
        self.slot_populated.fill_(True)
        self.private_ranks.fill_(self.max_private_lora_rank)
        self.factors_loaded.fill_(True)

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
        expected = {
            "shared_alpha": (self.block_count, self.shared_basis_count, self.width),
            "shared_beta": (self.block_count, self.width, self.shared_basis_count),
            "skill_coeffs": (self.max_skill_slots, self.shared_basis_count),
            "private_alpha": (self.block_count, self.max_skill_slots, self.max_private_lora_rank, self.width),
            "private_beta": (self.block_count, self.max_skill_slots, self.width, self.max_private_lora_rank),
            "anchors": (self.max_skill_slots, self.width),
        }
        tensors = {
            "shared_alpha": shared_alpha,
            "shared_beta": shared_beta,
            "skill_coeffs": skill_coeffs,
            "private_alpha": private_alpha,
            "private_beta": private_beta,
            "anchors": anchors,
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != expected[name]:
                raise ValueError(
                    f"{name} must have shape {expected[name]}, got {tuple(tensor.shape)}"
                )

        for name, param in [
            ("shared_alpha", self.shared_alpha),
            ("shared_beta", self.shared_beta),
            ("skill_coeffs", self.skill_coeffs),
            ("private_alpha", self.private_alpha),
            ("private_beta", self.private_beta),
            ("anchors", self.anchors),
        ]:
            tensor = tensors[name]
            param.copy_(
                tensor.to(device=param.device, dtype=param.dtype)
            )

        # Load slot population mask
        if slot_populated is not None:
            if tuple(slot_populated.shape) != (self.max_skill_slots,):
                raise ValueError(
                    f"slot_populated must have shape ({self.max_skill_slots},), got {tuple(slot_populated.shape)}"
                )
            self.slot_populated.copy_(
                slot_populated.to(device=self.slot_populated.device, dtype=torch.bool)
            )
        else:
            self.slot_populated.fill_(True)

        # Load per-skill private ranks
        if private_ranks is not None:
            if tuple(private_ranks.shape) != (self.max_skill_slots,):
                raise ValueError(
                    f"private_ranks must have shape ({self.max_skill_slots},), got {tuple(private_ranks.shape)}"
                )
            self.private_ranks.copy_(
                private_ranks.to(device=self.private_ranks.device, dtype=torch.long)
            )
        else:
            self.private_ranks.fill_(self.max_private_lora_rank)

        # Freeze anchors in production mode after loading fitted values.
        if self.init_mode == "production":
            self.anchors.requires_grad_(False)
        self.factors_loaded.fill_(True)

    def _assert_production_ready(self) -> None:
        """Raise if production mode assets are not loaded."""
        if self.init_mode == "production":
            if not bool(self.factors_loaded):
                raise RuntimeError(
                    "Production mode requires loaded skill factors. "
                    "Call load_skill_factors() before forward()."
                )

    def route(
        self,
        hidden: Tensor,
        *,
        block_index: int,
    ) -> RoutingContext:
        """Geometric skill routing: cosine similarity + signed top-k.

        Only populated skill slots participate in routing.  Unpopulated slots
        are masked out before top-k selection.

        Returns a RoutingContext carrying the skill mask, indices, and weights.
        This is called once per block visit and reused by all downstream
        consumers (branch contraction, working LoRA, expert soma).
        """
        self._assert_production_ready()
        scores = torch.einsum(
            "btd,sd->bts",
            F.normalize(hidden, dim=-1),
            F.normalize(self.anchors, dim=-1),
        )
        # Mask unpopulated slots: set their scores to 0 so they cannot be selected
        populated = self.slot_populated.to(scores.device, dtype=scores.dtype)  # [S]
        scores = scores * populated
        indices, selected_weights, dense_weights = _signed_topk(
            scores,
            top_k=self.top_k,
            epsilon=self.epsilon,
        )
        skill_mask = dense_weights != 0  # [B, T, max_skill_slots]
        return RoutingContext(
            skill_indices=indices,
            skill_weights=selected_weights,
            skill_dense_weights=dense_weights,
            skill_mask=skill_mask,
        )

    def apply_working_lora(
        self,
        hidden: Tensor,
        routing: RoutingContext,
        *,
        block_index: int,
    ) -> Tensor:
        """Apply the shared+private working LoRA: Δh = Σ_s g_{t,s} (ΔW_s^shared + X_s) h.

        Shared path (computed once per token, reused across all active skills):
            low_shared = a_j^T h          → [B*T, K]
            For each active skill s: q_{s,j} · low_shared → shared contribution

        Private path (gathered per active skill):
            low_priv = A_s^priv h          → [B*T, top_k, R_priv]
            B_s^priv · low_priv            → private contribution

        Both paths are scaled by routing weight g_{t,s} and summed.
        """
        self._assert_production_ready()
        batch, seq_len, _ = hidden.shape

        # Shared basis factors for this block
        sa = self.shared_alpha[block_index]   # [K, D]
        sb = self.shared_beta[block_index]     # [D, K]
        coeffs = self.skill_coeffs  # [S, K] — shared across blocks
        pa = self.private_alpha[block_index]   # [S, R_priv, D]
        pb = self.private_beta[block_index]     # [S, D, R_priv]

        # Flatten batch and sequence for efficient gathering
        flat_idx = routing.skill_indices.reshape(-1, self.top_k)  # [N, top_k]
        flat_hidden = hidden.reshape(-1, self.width)               # [N, D]
        flat_g = routing.skill_weights.reshape(-1, self.top_k)     # [N, top_k]
        N = flat_hidden.shape[0]

        # --- Shared path ---
        # low_shared: project hidden onto shared input factors → [N, K]
        low_shared = flat_hidden @ sa.T  # [N, K]

        # Gather per-active-skill coefficients: [N, top_k, K]
        coeffs_gathered = coeffs[flat_idx]  # [N, top_k, K]

        # Shared contribution per active skill: [N, top_k]
        # shared_skill = sum_j q_{s,j} * (a_j^T h) = q_s · low_shared
        shared_per_skill = torch.einsum("njk,nk->nj", coeffs_gathered, low_shared)  # [N, top_k]

        # Project shared scalar through shared output: b_j weighted by the scalar
        # But shared_per_skill is already a scalar per (token, skill) after contracting K.
        # The full shared update is: Σ_s g_s * (Σ_j q_{s,j} b_j (a_j^T h))
        # = Σ_s g_s * B_shared @ diag(q_s) @ A_shared @ h
        # Efficient: shared_proj = sb @ (q_s ⊙ low_shared) → [D] per skill
        # shared_weighted: [N, top_k, K] = q_s ⊙ low_shared (broadcast)
        shared_weighted = coeffs_gathered * low_shared.unsqueeze(1)  # [N, top_k, K]
        # Project through sb: [N, top_k, D] = shared_weighted @ sb^T
        shared_out = torch.einsum("njk,dk->njd", shared_weighted, sb)  # [N, top_k, D]
        # Scale by routing weight: [N, D] = sum_k g_k * shared_out_k
        shared_update = torch.einsum("nk,nkd->nd", flat_g, shared_out)  # [N, D]

        # --- Private path ---
        # Gather private factors for active skills
        pa_gathered = pa[flat_idx]  # [N, top_k, R_max, D]
        pb_gathered = pb[flat_idx]  # [N, top_k, D, R_max]

        # Per-skill rank mask: zero out padded private dimensions
        # Build a mask [S, R_max] where mask[s, r] = 1 if r < private_ranks[s]
        rank_range = torch.arange(self.max_private_lora_rank, device=pa.device)  # [R_max]
        rank_mask = (rank_range.unsqueeze(0) < self.private_ranks.unsqueeze(1)).to(pa.dtype)  # [S, R_max]
        # Gather mask for active skills: [N, top_k, R_max]
        rank_mask_gathered = rank_mask[flat_idx]  # [N, top_k, R_max]

        # Private low-rank projection: [N, top_k, R_max]
        low_priv = torch.einsum("nd,nkrd->nkr", flat_hidden, pa_gathered)
        # Mask padded ranks to zero
        low_priv = low_priv * rank_mask_gathered

        # Private output projection: [N, top_k, D]
        priv_out = torch.einsum("nkr,nkdr->nkd", low_priv, pb_gathered)

        # Scale by routing weight: [N, D]
        priv_update = torch.einsum("nk,nkd->nd", flat_g, priv_out)

        # --- Total ---
        update_flat = shared_update + priv_update
        return update_flat.view(batch, seq_len, self.width)

    def forward(
        self,
        hidden: Tensor,
        *,
        block_index: int,
    ) -> tuple[Tensor, RoutingContext]:
        """Route skills and apply composed working LoRA in one call."""
        routing = self.route(hidden, block_index=block_index)
        delta_h = self.apply_working_lora(hidden, routing, block_index=block_index)
        return delta_h, routing


class DendriticBranch(nn.Module):
    def __init__(self, width: int, hidden_width: int) -> None:
        super().__init__()
        self.content = nn.Linear(width, hidden_width)
        self.gate = nn.Linear(width, hidden_width)
        self.output = nn.Linear(hidden_width, width, bias=False)
        self.evidence = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.evidence, std=width**-0.5)
        nn.init.normal_(self.output.weight, std=1e-3 * hidden_width**-0.5)

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
        epsilon: float = 1e-6,
        deductive_branches_per_block: int = 0,
        max_premises: int = 8,
        max_contradictions: int = 4,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.width = int(width)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DendriticBranch(width, hidden_width)
                        for _ in range(branch_count)
                    ]
                )
                for _ in range(block_count)
            ]
        )
        # Typed deductive branch instances: one set per block.
        # These carry trainable conclusion/contradiction anchors.
        # The shared DeductiveBranchOperator is passed at runtime.
        self.deductive_branches = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DeductiveBranchInstance(
                            width,
                            max_premises=max_premises,
                            max_contradictions=max_contradictions,
                            epsilon=epsilon,
                        )
                        for _ in range(deductive_branches_per_block)
                    ]
                )
                for _ in range(block_count)
            ]
            if deductive_branches_per_block > 0
            else None
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

    def contract_deductive_branches(
        self,
        hidden: Tensor,
        *,
        block_index: int,
        operator: DeductiveBranchOperator,
    ) -> tuple[Tensor, list[BranchOutcome]] | None:
        """Run all deductive branches for this expert in the given block.

        Returns None if this expert has no deductive branches for this block.
        Otherwise returns (combined_movement, outcomes) where combined_movement
        is the signed-L1-normalized sum of gated branch movements.
        """
        if self.deductive_branches is None:
            return None
        branches = self.deductive_branches[block_index]
        if len(branches) == 0:
            return None

        movements = []
        outcomes = []
        for branch in branches:
            gated_movement, outcome = branch(hidden, operator)
            movements.append(gated_movement)
            outcomes.append(outcome)

        combined = combine_branch_movements(
            [m for m in movements],
            [o.evidence for o in outcomes],
            epsilon=self.epsilon,
        )
        return combined, outcomes


class ConditionalExpertBank(nn.Module):
    """High-dimensional experts reached through skill adjacency.

    Expert anchors must come from clustered residual trajectories in
    production. In smoke mode, random anchors serve structural testing.
    """

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
        init_mode: str = "smoke",
        epsilon: float = 1e-6,
        deductive_branches_per_block: int = 0,
        max_premises: int = 8,
        max_contradictions: int = 4,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.skill_count = int(skill_count)
        self.expert_count = int(expert_count)
        self.top_k = int(top_k)
        self.init_mode = str(init_mode)
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
                    epsilon=epsilon,
                    deductive_branches_per_block=deductive_branches_per_block,
                    max_premises=max_premises,
                    max_contradictions=max_contradictions,
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
        self.register_buffer(
            "expert_registry_loaded",
            torch.tensor(init_mode == "smoke", dtype=torch.bool),
        )

    def _assert_production_ready(self) -> None:
        if self.init_mode == "production" and not bool(self.expert_registry_loaded):
            raise RuntimeError(
                "Production mode requires loaded expert registry. "
                "Call load_expert_registry() before forward()."
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
            adjacency: [skill_count, expert_count] bool mask.
        """
        if tuple(anchors.shape) != (self.expert_count, self.width):
            raise ValueError(
                f"Expert anchors must have shape ({self.expert_count}, {self.width})"
            )
        expected_adj = (self.skill_count, self.expert_count)
        if tuple(adjacency.shape) != expected_adj:
            raise ValueError(f"Expert adjacency must have shape {expected_adj}")
        value = adjacency.to(
            device=self.skill_adjacency.device,
            dtype=torch.bool,
        )
        if not value.any(dim=-1).all():
            raise ValueError("Every skill requires at least one adjacent expert")
        self.anchors.copy_(
            anchors.to(device=self.anchors.device, dtype=self.anchors.dtype)
        )
        self.skill_adjacency.copy_(value)
        self.expert_registry_loaded.fill_(True)

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

    def select_experts(
        self,
        hidden: Tensor,
        *,
        routing: RoutingContext,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Route to experts through skill adjacency from a RoutingContext."""
        self._assert_production_ready()
        active_skill_indices = routing.skill_indices
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
        return indices, selected_weights, dense_weights

    def forward(
        self,
        hidden: Tensor,
        *,
        block_index: int,
        routing: RoutingContext,
    ) -> tuple[Tensor, DirectionRouting]:
        indices, selected_weights, dense_weights = self.select_experts(
            hidden, routing=routing
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

    def contract_branches(
        self,
        hidden: Tensor,
        *,
        block_index: int,
        routing: RoutingContext,
        operator: DeductiveBranchOperator,
    ) -> tuple[Tensor, BranchContractionStats | None]:
        """Run typed branch contraction for all active experts (first sublayer).

        Follows the spec dependency chain:
            skills -> skill-to-expert adjacency -> selected experts -> branch specs

        For each selected expert that carries deductive branches, instantiate
        those branches against the current hidden state, run the shared
        DeductiveBranchOperator, and combine movements with signed L1 norm.

        Returns (combined_branch_movement, stats) where the movement is the
        block-soma combination of all expert branch movements.
        """
        self._assert_production_ready()
        indices, _, dense_weights = self.select_experts(
            hidden, routing=routing
        )

        # Collect branch movements from all active experts that have deductive branches
        expert_movements = []  # list of [B, T, D] per expert
        expert_activations = []  # list of [B, T] expert weights for soma
        all_evidences = []  # flat list across all experts' branches
        all_residuals = []

        for expert_id in torch.unique(indices).tolist():
            expert = self.experts[expert_id]
            result = expert.contract_deductive_branches(
                hidden, block_index=block_index, operator=operator
            )
            if result is None:
                continue
            combined, outcomes = result
            # Expert activation from routing weights
            expert_weight = dense_weights[..., expert_id]  # [B, T]
            expert_movements.append(combined)
            expert_activations.append(expert_weight)
            for outcome in outcomes:
                all_evidences.append(outcome.evidence)
                all_residuals.append(outcome.residual)

        if not expert_movements:
            # No active expert has deductive branches
            return torch.zeros_like(hidden), None

        # Block soma: combine expert movements with signed L1 norm
        # Delta_soma = sum(a_e * Delta_e) / (epsilon + sum(|a_e|))
        stacked_movements = torch.stack(expert_movements, dim=-1)  # [B, T, D, E]
        stacked_activations = torch.stack(expert_activations, dim=-1)  # [B, T, E]
        abs_sum = stacked_activations.abs().sum(dim=-1, keepdim=True)  # [B, T, 1]
        soma_weights = stacked_activations / (abs_sum + self.epsilon)  # [B, T, E]
        branch_movement = (stacked_movements * soma_weights.unsqueeze(-2)).sum(dim=-1)  # [B, T, D]

        stats = BranchContractionStats(
            movement=branch_movement,
            branch_count=len(all_evidences),
            branch_evidences=tuple(all_evidences),
            branch_residuals=tuple(all_residuals),
        )
        return branch_movement, stats


class SkillExpertSystem(nn.Module):
    """Composed working-LoRA skill layer + conditional expert hierarchy.

    Routes skills once per block visit, then reuses that RoutingContext for:
    - branch contraction (first sublayer)
    - composed working LoRA (second sublayer)
    - expert soma (second sublayer)
    """

    def __init__(
        self,
        width: int,
        *,
        max_skill_slots: int,
        shared_basis_count: int,
        max_private_lora_rank: int,
        skill_top_k: int,
        expert_count: int,
        expert_hidden_width: int,
        expert_branches: int,
        expert_top_k: int,
        init_mode: str = "smoke",
        epsilon: float = 1e-6,
        deductive_branches_per_block: int = 0,
        max_premises: int = 8,
        max_contradictions: int = 4,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.epsilon = float(epsilon)
        self.init_mode = str(init_mode)
        self.skills = ComposedWorkingLoRA(
            width,
            max_skill_slots=max_skill_slots,
            shared_basis_count=shared_basis_count,
            max_private_lora_rank=max_private_lora_rank,
            top_k=skill_top_k,
            init_mode=init_mode,
            epsilon=epsilon,
        )
        self.experts = ConditionalExpertBank(
            width,
            skill_count=max_skill_slots,
            expert_count=expert_count,
            hidden_width=expert_hidden_width,
            branch_count=expert_branches,
            top_k=expert_top_k,
            init_mode=init_mode,
            epsilon=epsilon,
            deductive_branches_per_block=deductive_branches_per_block,
            max_premises=max_premises,
            max_contradictions=max_contradictions,
        )
        # Shared deductive operator: one instance serves all experts
        self.deductive_operator = (
            DeductiveBranchOperator(
                width,
                harmonic_exponent=2.0,
                epsilon=epsilon,
            )
            if deductive_branches_per_block > 0
            else None
        )
        self.skill_gate = nn.Parameter(torch.tensor(1e-3))
        self.expert_gate = nn.Parameter(torch.tensor(1e-3))

    def route(
        self,
        hidden: Tensor,
        *,
        block_index: int,
    ) -> RoutingContext:
        """Route skills once per block visit. Returns RoutingContext for reuse."""
        return self.skills.route(hidden, block_index=block_index)

    def contract_branches(
        self,
        hidden: Tensor,
        *,
        block_index: int,
        routing: RoutingContext,
    ) -> tuple[Tensor, BranchContractionStats | None]:
        """First sublayer: branch contraction using the established RoutingContext.

        Returns (branch_movement, stats) for the contraction residual.
        If no deductive branches are configured, returns (zeros, None).
        """
        if self.deductive_operator is None:
            return torch.zeros_like(hidden), None

        branch_movement, stats = self.experts.contract_branches(
            hidden,
            block_index=block_index,
            routing=routing,
            operator=self.deductive_operator,
        )
        return branch_movement, stats

    def forward(
        self,
        hidden: Tensor,
        *,
        block_index: int,
        routing: RoutingContext | None = None,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, SkillExpertStats]:
        """Second sublayer: composed working LoRA + expert soma.

        If routing is None, routes skills internally. Otherwise reuses
        the provided RoutingContext (the normal path from recurrent_core
        which routes once and passes the context through).
        """
        if routing is None:
            routing = self.route(hidden, block_index=block_index)

        skill_update = self.skills.apply_working_lora(
            hidden, routing, block_index=block_index
        )
        expert_update, expert_route = self.experts(
            hidden,
            block_index=block_index,
            routing=routing,
        )
        update = (
            torch.tanh(self.skill_gate) * skill_update
            + torch.tanh(self.expert_gate) * expert_update
        )
        if return_stats:
            return update, SkillExpertStats(
                skills=DirectionRouting(routing.skill_indices, routing.skill_weights),
                experts=expert_route,
                skill_gate=torch.tanh(self.skill_gate),
                expert_gate=torch.tanh(self.expert_gate),
            )
        return update
