"""Two stored physical blocks reused to carry thought across loop rounds."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor, nn
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.recurrent_core requires PyTorch. Install torch>=2.7."
    ) from error

from .config import DendritronConfig
from .geometric_attention import (
    HarMaxContraction,
    HarMaxContractionStats,
    RMSNorm,
)
from .definition_lngram import DefinitionLnGram, DefinitionLnGramStats
from .lngram import LNGramMemory, LNGramStats
from .memory_fusion import MemoryFusionStats, MemoryPayloads, SparseMemoryFusion
from .working_adapter import BranchContractionStats, SkillExpertStats, SkillExpertSystem


@dataclass(frozen=True)
class BlockVisitStats:
    round_index: int
    block_index: int
    contraction_relative_change: Tensor
    compute_relative_change: Tensor
    relative_change: Tensor
    harmax: HarMaxContractionStats
    memory: MemoryFusionStats
    lngram: LNGramStats | None
    skill_expert: SkillExpertStats
    definition_lngram: DefinitionLnGramStats | None = None
    branch_contraction: BranchContractionStats | None = None


@dataclass(frozen=True)
class RecurrentCoreStats:
    visits: tuple[BlockVisitStats, ...]
    rounds_executed: int
    final_relative_change: Tensor
    alpha: float
    beta: float


def _relative_change(current: Tensor, previous: Tensor, epsilon: float) -> Tensor:
    numerator = (current - previous).square().mean(dim=(-2, -1)).sqrt()
    denominator = previous.square().mean(dim=(-2, -1)).sqrt()
    return numerator / (denominator + float(epsilon))


class TwoBlockRecurrentCore(nn.Module):
    """The live thought state is ``hidden``; both blocks repeatedly update it.

    Every physical block has two sequential Post-RMSNorm residual sublayers:
    HarMax/memory/concept contraction followed by shared-skill/expert compute.
    """

    def __init__(
        self,
        config: DendritronConfig,
        *,
        memory_fusion: SparseMemoryFusion,
    ) -> None:
        super().__init__()
        self.config = config
        self.memory_fusion = memory_fusion

        # N is the block-equivalent unrolled depth. Beta remains the DeepLoop
        # initialization contract for designated trainable residual matrices;
        # the Euclidean HarMax derivative itself has no learned metric matrix.
        effective_depth = 2 * config.loop_rounds
        exponent = config.deep_loop_exponent
        self.alpha = float((2.0 * effective_depth) ** exponent)
        self.beta = float((8.0 * effective_depth) ** (-exponent))

        self.harmax = nn.ModuleList(
            [
                HarMaxContraction(
                    config.model_width,
                    max_sequence_length=config.max_sequence_length,
                    candidate_window=config.context_window,
                    top_k=config.context_top_k,
                    harmonic_exponent=config.harmax_exponent,
                    epsilon=config.geometric_epsilon,
                )
                for _ in range(2)
            ]
        )
        self.contraction_norms = nn.ModuleList(
            [
                RMSNorm(config.model_width, config.residual_epsilon)
                for _ in range(2)
            ]
        )
        self.compute_norms = nn.ModuleList(
            [
                RMSNorm(config.model_width, config.residual_epsilon)
                for _ in range(2)
            ]
        )
        self.lngram = (
            nn.ModuleList(
                [
                    LNGramMemory(
                        config.model_width,
                        bits_per_route=config.lngram_bits_per_route,
                        orders=config.lngram_orders,
                        route_memory_width=config.lngram_route_memory_width,
                        readout_mode="distance",
                        readout_epsilon=config.geometric_epsilon,
                    )
                    for _ in range(2)
                ]
            )
            if config.use_lngram
            else None
        )
        # Definition LNGram: routes in joint space, gathers from frozen bank,
        # applies signed HarMax (y-p).  Requires attach_definition_bank
        # before forward.  When the bank is attached, this replaces the
        # learned-table LNGram in the forward pass.
        self.definition_lngram = (
            nn.ModuleList(
                [
                    DefinitionLnGram(
                        config.model_width,
                        config.memory_width,
                        bits_per_route=config.lngram_bits_per_route,
                        orders=config.lngram_orders,
                        senses_per_address=config.senses_per_address,
                        harmonic_exponent=config.harmax_exponent,
                        epsilon=config.geometric_epsilon,
                    )
                    for _ in range(2)
                ]
            )
            if config.use_lngram
            else None
        )
        self.skill_expert = SkillExpertSystem(
            config.model_width,
            max_skill_slots=config.max_skill_slots,
            shared_basis_count=config.shared_basis_count,
            max_private_lora_rank=config.max_private_lora_rank,
            skill_top_k=config.skill_top_k,
            expert_count=config.expert_count,
            expert_hidden_width=config.expert_hidden_width,
            expert_branches=config.expert_branches,
            expert_top_k=config.expert_top_k,
            init_mode=config.init_mode,
            epsilon=config.geometric_epsilon,
            deductive_branches_per_block=config.deductive_branches_per_block,
            max_premises=config.deductive_max_premises,
            max_contradictions=config.deductive_max_contradictions,
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        memory_payloads: MemoryPayloads | None = None,
        rounds: int | None = None,
        adaptive_threshold: float | None = None,
        minimum_rounds: int = 1,
        return_stats: bool = False,
    ) -> Tensor | tuple[Tensor, RecurrentCoreStats]:
        total_rounds = self.config.loop_rounds if rounds is None else int(rounds)
        if total_rounds < 1:
            raise ValueError("rounds must be positive")
        if minimum_rounds < 1 or minimum_rounds > total_rounds:
            raise ValueError("minimum_rounds must fall inside the round budget")

        visits: list[BlockVisitStats] = []
        final_change = hidden.new_full(hidden.shape[:1], float("inf"))
        rounds_executed = 0
        for round_index in range(total_rounds):
            for block_index in range(2):
                block_input = hidden
                harmax_update, harmax_stats = self.harmax[block_index](
                    hidden,
                    return_stats=True,
                )

                # Memory locality is evaluated after causal context has moved
                # the query, while all definition points remain available.
                contextual_hidden = hidden + harmax_update
                memory_update, memory_stats = self.memory_fusion(
                    contextual_hidden,
                    memory_payloads,
                    block_index=block_index,
                    return_stats=True,
                )
                # Use DefinitionLnGram when the frozen bank is attached;
                # fall back to learned-table LNGram otherwise.
                definition_lngram_stats = None
                if (
                    self.definition_lngram is not None
                    and self.definition_lngram[block_index].definition_bank is not None
                ):
                    lngram_input = contextual_hidden + memory_update
                    lngram_output, definition_lngram_stats = (
                        self.definition_lngram[block_index](
                            lngram_input,
                            return_stats=True,
                        )
                    )
                    lngram_update = lngram_output - lngram_input
                    lngram_stats = None
                elif self.lngram is None:
                    lngram_update = torch.zeros_like(hidden)
                    lngram_stats = None
                else:
                    lngram_input = contextual_hidden + memory_update
                    lngram_output, lngram_stats = self.lngram[block_index](
                        lngram_input,
                        return_stats=True,
                    )
                    lngram_update = lngram_output - lngram_input

                # Branch contraction: route skills once -> select experts ->
                # instantiate deductive branches -> HarMax contraction.
                # Uses the post-LNGram state as input, following the spec's
                # block order: memory + LNGram -> branch contraction -> first RMSNorm.
                branch_input = contextual_hidden + memory_update + lngram_update
                # Route once: this RoutingContext is reused for branch contraction
                # and the second-sublayer skill-expert compute.
                routing = self.skill_expert.route(
                    branch_input, block_index=block_index
                )
                branch_movement, branch_stats = self.skill_expert.contract_branches(
                    branch_input,
                    block_index=block_index,
                    routing=routing,
                )

                contraction_residual = (
                    harmax_update + memory_update + lngram_update + branch_movement
                )
                contracted = self.contraction_norms[block_index](
                    self.alpha * hidden + contraction_residual
                )
                contraction_change = _relative_change(
                    contracted,
                    hidden,
                    self.config.residual_epsilon,
                )

                skill_expert, skill_expert_stats = self.skill_expert(
                    contracted,
                    block_index=block_index,
                    routing=routing,
                    return_stats=True,
                )
                hidden = self.compute_norms[block_index](
                    self.alpha * contracted + skill_expert
                )
                compute_change = _relative_change(
                    hidden,
                    contracted,
                    self.config.residual_epsilon,
                )
                relative_change = _relative_change(
                    hidden,
                    block_input,
                    self.config.residual_epsilon,
                )
                final_change = relative_change
                if return_stats:
                    visits.append(
                        BlockVisitStats(
                            round_index=round_index,
                            block_index=block_index,
                            contraction_relative_change=contraction_change,
                            compute_relative_change=compute_change,
                            relative_change=relative_change,
                            harmax=harmax_stats,
                            memory=memory_stats,
                            lngram=lngram_stats,
                            definition_lngram=definition_lngram_stats,
                            branch_contraction=branch_stats,
                            skill_expert=skill_expert_stats,
                        )
                    )
            rounds_executed = round_index + 1
            if (
                adaptive_threshold is not None
                and rounds_executed >= minimum_rounds
                and bool((final_change < adaptive_threshold).all())
            ):
                break

        if return_stats:
            return hidden, RecurrentCoreStats(
                visits=tuple(visits),
                rounds_executed=rounds_executed,
                final_relative_change=final_change,
                alpha=self.alpha,
                beta=self.beta,
            )
        return hidden
