"""Verified transition records for procedural SVD input.

Each record carries the full provenance of one successful live-state
transition through the recurrent core:

    (h_t, h_{t+1}, delta_h_t, task_id, trajectory_id, skill_ids,
     block_index, round_index, success_evidence, ordering)

D-065 extensions over the original D-063 schema:
  - task_id and trajectory_id identify the episode.
  - skill_ids is optional observed-routing metadata; it may be empty
    during initial skill discovery because SVD discovers skill slots
    after capture.
  - A structured success_evidence payload authorizes the derived
    success verdict; success is a read-only property of that evidence.
  - ordering is scoped within each trajectory, not globally monotonic.
  - All tensor snapshots are detached clones that release the runtime
    computation graph.
  - delta_matrix() is the canonical procedural-SVD input; h_matrix()
    remains a diagnostic view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    import torch
    from torch import Tensor
except ImportError as error:  # pragma: no cover - exercised on training host
    raise ImportError(
        "dendritron.transition_records requires PyTorch. Install torch>=2.7."
    ) from error


# ---------------------------------------------------------------------------
# Structured success evidence (D-065)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuccessEvidence:
    """Structured payload that authorizes the derived success verdict.

    The verdict field stores the boolean outcome.  The reason field
    carries a human-readable explanation.  The metrics dict holds
    quantitative evidence (loss, accuracy, rank margin, etc.) that
    justifies the verdict.  success is a read-only property derived
    from verdict so that the evidence remains the single source of truth.
    """

    verdict: bool
    reason: str = ""
    metrics: Mapping[str, float] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Derived success verdict — the evidence authorizes this."""
        return self.verdict

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "metrics": dict(self.metrics),
        }


# ---------------------------------------------------------------------------
# Transition record (D-063 + D-065)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TransitionRecord:
    """One verified live-state transition with full provenance.

    Fields:
        h_t:              [D] hidden state before the transition (detached clone).
        h_next:           [D] hidden state after the transition (detached clone).
        delta_h:          [D] = h_next - h_t (detached clone).
        task_id:          Identifier for the task that produced this transition.
        trajectory_id:    Identifier for the trajectory within the task.
        skill_ids:        tuple of active skill slot indices; may be empty
                          during initial skill discovery before SVD assigns slots.
        block_index:      physical block index (0 or 1).
        round_index:      recurrent round index (0-based).
        success_evidence: structured payload authorizing the success verdict.
        ordering:         int, sequence number scoped within each trajectory.
    """

    h_t: Tensor
    h_next: Tensor
    delta_h: Tensor
    task_id: str
    trajectory_id: str
    skill_ids: tuple[int, ...]
    block_index: int
    round_index: int
    success_evidence: SuccessEvidence
    ordering: int

    # --- Derived properties ---

    @property
    def success(self) -> bool:
        """Derived from success_evidence.verdict — not stored independently."""
        return self.success_evidence.success

    # --- Validation ---

    def __post_init__(self) -> None:
        if self.h_t.shape != self.h_next.shape:
            raise ValueError(
                f"h_t shape {tuple(self.h_t.shape)} != "
                f"h_next shape {tuple(self.h_next.shape)}"
            )
        if self.delta_h.shape != self.h_t.shape:
            raise ValueError(
                f"delta_h shape {tuple(self.delta_h.shape)} != "
                f"h_t shape {tuple(self.h_t.shape)}"
            )
        if self.block_index not in (0, 1):
            raise ValueError(f"block_index must be 0 or 1, got {self.block_index}")
        if self.round_index < 0:
            raise ValueError(f"round_index must be non-negative, got {self.round_index}")
        if self.ordering < 0:
            raise ValueError(f"ordering must be non-negative, got {self.ordering}")
        if not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if not self.trajectory_id:
            raise ValueError("trajectory_id must be a non-empty string")
        # skill_ids may be empty — D-065 allows initial discovery without slots
        for s in self.skill_ids:
            if s < 0:
                raise ValueError(f"skill_id {s} must be non-negative")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class TransitionRecordBuilder:
    """Collects per-visit hidden states and emits verified transition records.

    Ordering is scoped per trajectory: each unique trajectory_id maintains
    its own monotonic counter.  All tensor snapshots are detached clones
    that release the runtime computation graph.

    Usage:
        builder = TransitionRecordBuilder()
        builder.add_visit(
            h_before, h_after,
            task_id="arithmetic", trajectory_id="traj-001",
            skill_ids=(0, 3), block_index=0, round_index=0,
            success_evidence=SuccessEvidence(
                verdict=True, reason="correct sum",
                metrics={"loss": 0.01},
            ),
        )
        records = builder.build()
        delta_matrix = builder.delta_matrix()  # [N, D] for SVD
    """

    def __init__(self) -> None:
        self._records: list[TransitionRecord] = []
        # Per-trajectory ordering counters (D-065: scoped within trajectory)
        self._trajectory_counters: dict[str, int] = {}

    def add_visit(
        self,
        h_t: Tensor,
        h_next: Tensor,
        *,
        task_id: str = "default",
        trajectory_id: str = "default",
        skill_ids: Sequence[int] = (),
        block_index: int = 0,
        round_index: int = 0,
        success_evidence: SuccessEvidence | None = None,
        success: bool | None = None,
    ) -> TransitionRecord:
        """Record one transition and return the constructed record.

        delta_h is computed as h_next - h_t.  All three tensors are
        detached and cloned so the record is independent of the runtime
        computation graph.

        Args:
            h_t:              [D] pre-transition hidden state.
            h_next:           [D] post-transition hidden state.
            task_id:          Task identifier.
            trajectory_id:    Trajectory identifier within the task.
            skill_ids:        Active skill slot indices; may be empty
                              during initial skill discovery.
            block_index:      Physical block index (0 or 1).
            round_index:      Recurrent round index (0-based).
            success_evidence: Structured evidence payload.  If None and
                              success is provided, a minimal SuccessEvidence
                              is constructed for backward compatibility.
            success:          Boolean verdict for backward compatibility.
                              Ignored if success_evidence is provided.
        """
        if h_t.ndim != 1:
            raise ValueError(
                f"h_t must be 1D [D], got shape {tuple(h_t.shape)}"
            )
        if h_next.shape != h_t.shape:
            raise ValueError(
                f"h_next shape {tuple(h_next.shape)} != "
                f"h_t shape {tuple(h_t.shape)}"
            )

        # Resolve success evidence (D-065: structured payload is primary)
        if success_evidence is not None:
            evidence = success_evidence
        elif success is not None:
            evidence = SuccessEvidence(
                verdict=bool(success),
                reason="backward-compatible boolean verdict",
            )
        else:
            evidence = SuccessEvidence(verdict=True, reason="default success")

        # Per-trajectory ordering (D-065: scoped within trajectory)
        ordering = self._trajectory_counters.get(trajectory_id, 0)
        self._trajectory_counters[trajectory_id] = ordering + 1

        # Detached clones release the runtime computation graph (D-065)
        delta = (h_next - h_t).detach().clone()
        record = TransitionRecord(
            h_t=h_t.detach().clone(),
            h_next=h_next.detach().clone(),
            delta_h=delta,
            task_id=str(task_id),
            trajectory_id=str(trajectory_id),
            skill_ids=tuple(int(s) for s in skill_ids),
            block_index=int(block_index),
            round_index=int(round_index),
            success_evidence=evidence,
            ordering=ordering,
        )
        self._records.append(record)
        return record

    def build(
        self,
        *,
        verified_only: bool = True,
        task_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> list[TransitionRecord]:
        """Return records, optionally filtering by verification, task, or trajectory.

        Args:
            verified_only: If True, return only records with success=True.
            task_id:        If provided, filter to this task.
            trajectory_id:  If provided, filter to this trajectory.
        """
        records = list(self._records)
        if verified_only:
            records = [r for r in records if r.success]
        if task_id is not None:
            records = [r for r in records if r.task_id == task_id]
        if trajectory_id is not None:
            records = [r for r in records if r.trajectory_id == trajectory_id]
        return records

    def delta_matrix(
        self,
        *,
        verified_only: bool = True,
        task_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> Tensor:
        """Stack delta vectors into a [N, D] matrix for procedural SVD.

        This is the canonical procedural-SVD input (D-065).  Only verified
        records contribute.  Returns an empty [0, D] tensor if no records
        pass the filter.
        """
        records = self.build(
            verified_only=verified_only,
            task_id=task_id,
            trajectory_id=trajectory_id,
        )
        if not records:
            if self._records:
                return torch.zeros(0, self._records[0].h_t.shape[0])
            return torch.zeros(0, 0)
        return torch.stack([r.delta_h for r in records], dim=0)

    def h_matrix(
        self,
        *,
        verified_only: bool = True,
        task_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> Tensor:
        """Stack pre-transition h_t vectors into a [N, D] diagnostic matrix.

        This is a diagnostic view only (D-065).  Procedural SVD consumes
        delta_matrix(), not h_matrix().
        """
        records = self.build(
            verified_only=verified_only,
            task_id=task_id,
            trajectory_id=trajectory_id,
        )
        if not records:
            if self._records:
                return torch.zeros(0, self._records[0].h_t.shape[0])
            return torch.zeros(0, 0)
        return torch.stack([r.h_t for r in records], dim=0)

    def trajectories(self) -> list[str]:
        """Return distinct trajectory_ids that have at least one record."""
        return list(dict.fromkeys(r.trajectory_id for r in self._records))

    def tasks(self) -> list[str]:
        """Return distinct task_ids that have at least one record."""
        return list(dict.fromkeys(r.task_id for r in self._records))

    @property
    def count(self) -> int:
        """Total number of records (including unverified)."""
        return len(self._records)

    @property
    def verified_count(self) -> int:
        """Number of verified (success=True) records."""
        return sum(1 for r in self._records if r.success)
