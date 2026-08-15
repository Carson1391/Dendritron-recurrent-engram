"""Expert junction records connecting knowledge, tasks, skills, and branches.

An expert is a many-to-many routing junction. Its identity comes from a
knowledge anchor, a task relation, and edges to principal skill directions.
Episode-local Shared-LoRA coefficients remain separate fast state.

An expert may reference an optional coefficient prior for initialization. That
reference is metadata attached to the junction; it does not redefine the expert
as an adapter or coefficient tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class BranchSpec:
    branch_id: str
    operator: str
    relation: str
    skill_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.branch_id or not self.operator:
            raise ValueError("branch_id and operator are required")


@dataclass(frozen=True)
class ExpertRecord:
    expert_id: str
    knowledge_anchor: tuple[float, ...]
    task_relation: str
    skill_ids: tuple[int, ...]
    concept_ids: tuple[str, ...] = ()
    branches: tuple[BranchSpec, ...] = ()
    coefficient_prior_id: str | None = None
    success_count: int = 0
    metadata: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.expert_id:
            raise ValueError("expert_id is required")
        if not self.knowledge_anchor:
            raise ValueError("knowledge_anchor cannot be empty")
        if not self.task_relation:
            raise ValueError("task_relation is required")
        if len(set(self.skill_ids)) != len(self.skill_ids):
            raise ValueError("skill_ids must be unique")
        if any(skill_id < 0 for skill_id in self.skill_ids):
            raise ValueError("skill_ids must be nonnegative")
        if self.success_count < 0:
            raise ValueError("success_count must be nonnegative")


@dataclass(frozen=True)
class ExpertCandidate:
    expert: ExpertRecord
    matched_skill_ids: tuple[int, ...]
    matched_concept_ids: tuple[str, ...]
    task_relation_match: bool

    @property
    def score(self) -> tuple[int, int, int, int]:
        return (
            int(self.task_relation_match),
            len(self.matched_concept_ids),
            len(self.matched_skill_ids),
            self.expert.success_count,
        )


@dataclass
class ExpertGraph:
    """Appendable many-to-many adjacency for local expert routing."""

    experts: dict[str, ExpertRecord] = field(default_factory=dict)
    skill_to_experts: dict[int, set[str]] = field(default_factory=dict)
    concept_to_experts: dict[str, set[str]] = field(default_factory=dict)

    def add(self, expert: ExpertRecord) -> None:
        if expert.expert_id in self.experts:
            raise ValueError(f"Duplicate expert_id: {expert.expert_id}")
        self.experts[expert.expert_id] = expert
        for skill_id in expert.skill_ids:
            self.skill_to_experts.setdefault(skill_id, set()).add(expert.expert_id)
        for concept_id in expert.concept_ids:
            self.concept_to_experts.setdefault(concept_id, set()).add(expert.expert_id)

    def adjacent_to_skill(self, skill_id: int) -> tuple[ExpertRecord, ...]:
        return tuple(
            self.experts[expert_id]
            for expert_id in sorted(self.skill_to_experts.get(skill_id, ()))
        )

    def route(
        self,
        *,
        active_skill_ids: Iterable[int],
        active_concept_ids: Iterable[str] = (),
        task_relation: str | None = None,
        limit: int = 8,
    ) -> tuple[ExpertCandidate, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        skills = frozenset(int(value) for value in active_skill_ids)
        concepts = frozenset(str(value) for value in active_concept_ids)
        candidate_ids: set[str] = set()
        for skill_id in skills:
            candidate_ids.update(self.skill_to_experts.get(skill_id, ()))
        for concept_id in concepts:
            candidate_ids.update(self.concept_to_experts.get(concept_id, ()))

        candidates = []
        for expert_id in candidate_ids:
            expert = self.experts[expert_id]
            candidates.append(
                ExpertCandidate(
                    expert=expert,
                    matched_skill_ids=tuple(sorted(skills.intersection(expert.skill_ids))),
                    matched_concept_ids=tuple(
                        sorted(concepts.intersection(expert.concept_ids))
                    ),
                    task_relation_match=bool(
                        task_relation is not None
                        and expert.task_relation == task_relation
                    ),
                )
            )
        candidates.sort(
            key=lambda item: (
                *(-value for value in item.score),
                item.expert.expert_id,
            )
        )
        return tuple(candidates[:limit])
