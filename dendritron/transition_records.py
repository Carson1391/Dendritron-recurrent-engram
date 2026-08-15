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

D-066 correction:
  - A trajectory is identified by the ordered pair (task_id, trajectory_id).
    Per-trajectory counters, filtering, listing, and uniqueness use that
    pair so different tasks may safely reuse local names such as traj-001.
  - The canonical verified procedural-SVD path requires explicit task_id,
    trajectory_id, and structured success_evidence.  Compatibility defaults
    (bare success= bool, default task/trajectory IDs) are tagged as legacy
    input and excluded from the canonical SVD entry point.
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
# Legacy tag (D-066)
# ---------------------------------------------------------------------------

# Sentinel values that mark compatibility-default inputs as legacy.
# The canonical procedural-SVD path rejects records carrying these tags.
_LEGACY_TASK_ID = "__legacy_default_task__"
_LEGACY_TRAJECTORY_ID = "__legacy_default_trajectory__"


def _is_legacy_record(record: TransitionRecord) -> bool:
    """Check whether a record was created with legacy compatibility defaults."""
    return (
        record.task_id == _LEGACY_TASK_ID
        or record.trajectory_id == _LEGACY_TRAJECTORY_ID
        or record.success_evidence.reason == "backward-compatible boolean verdict"
        or record.success_evidence.reason == "default success"
    )


# ---------------------------------------------------------------------------
# Transition record (D-063 + D-065 + D-066)
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
        ordering:         int, sequence number scoped within each trajectory
                          identified by the pair (task_id, trajectory_id).
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

    @property
    def trajectory_key(self) -> tuple[str, str]:
        """The (task_id, trajectory_id) pair that uniquely identifies this trajectory."""
        return (self.task_id, self.trajectory_id)

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

    Ordering is scoped per trajectory identified by the pair
    (task_id, trajectory_id) (D-066).  Each unique pair maintains its own
    monotonic counter.  All tensor snapshots are detached clones that
    release the runtime computation graph.

    Two usage modes:

    **Canonical (D-066):** explicit task_id, trajectory_id, and
    structured success_evidence.  Use ``canonical_delta_matrix()`` to
    obtain the procedural-SVD input; it rejects legacy records.

    **Legacy (backward-compatible):** bare ``success=True/False`` with
    default task/trajectory IDs.  These records are tagged as legacy and
    excluded from the canonical SVD path.  Use ``build()`` and
    ``delta_matrix()`` for backward-compatible access.

    Usage (canonical):
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
        delta_matrix = builder.canonical_delta_matrix()  # [N, D] for SVD
    """

    def __init__(self) -> None:
        self._records: list[TransitionRecord] = []
        # Per-trajectory ordering counters keyed by (task_id, trajectory_id) (D-066)
        self._trajectory_counters: dict[tuple[str, str], int] = {}

    def add_visit(
        self,
        h_t: Tensor,
        h_next: Tensor,
        *,
        task_id: str = _LEGACY_TASK_ID,
        trajectory_id: str = _LEGACY_TRAJECTORY_ID,
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
            task_id:          Task identifier.  Defaults to legacy sentinel.
            trajectory_id:    Trajectory identifier within the task.
                              Defaults to legacy sentinel.
            skill_ids:        Active skill slot indices; may be empty
                              during initial skill discovery.
            block_index:      Physical block index (0 or 1).
            round_index:      Recurrent round index (0-based).
            success_evidence: Structured evidence payload (canonical D-066).
                              If None and success is provided, a minimal
                              SuccessEvidence is constructed and tagged
                              as legacy.
            success:          Boolean verdict for backward compatibility.
                              Ignored if success_evidence is provided.
                              Legacy tag is applied when this path is used.
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

        # Resolve success evidence and determine legacy status (D-066)
        is_legacy = False
        if success_evidence is not None:
            evidence = success_evidence
        elif success is not None:
            evidence = SuccessEvidence(
                verdict=bool(success),
                reason="backward-compatible boolean verdict",
            )
            is_legacy = True
        else:
            evidence = SuccessEvidence(verdict=True, reason="default success")
            is_legacy = True

        # Tag legacy task/trajectory IDs (D-066)
        effective_task_id = task_id
        effective_trajectory_id = trajectory_id
        if is_legacy and task_id == _LEGACY_TASK_ID:
            effective_task_id = _LEGACY_TASK_ID
        if is_legacy and trajectory_id == _LEGACY_TRAJECTORY_ID:
            effective_trajectory_id = _LEGACY_TRAJECTORY_ID

        # Per-trajectory ordering keyed by (task_id, trajectory_id) pair (D-066)
        pair_key = (effective_task_id, effective_trajectory_id)
        ordering = self._trajectory_counters.get(pair_key, 0)
        self._trajectory_counters[pair_key] = ordering + 1

        # Detached clones release the runtime computation graph (D-065)
        delta = (h_next - h_t).detach().clone()
        record = TransitionRecord(
            h_t=h_t.detach().clone(),
            h_next=h_next.detach().clone(),
            delta_h=delta,
            task_id=effective_task_id,
            trajectory_id=effective_trajectory_id,
            skill_ids=tuple(int(s) for s in skill_ids),
            block_index=int(block_index),
            round_index=int(round_index),
            success_evidence=evidence,
            ordering=ordering,
        )
        self._records.append(record)
        return record

    # -----------------------------------------------------------------------
    # Backward-compatible build / matrix access (includes legacy records)
    # -----------------------------------------------------------------------

    def build(
        self,
        *,
        verified_only: bool = True,
        task_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> list[TransitionRecord]:
        """Return records, optionally filtering by verification, task, or trajectory.

        When both task_id and trajectory_id are provided, filtering uses
        the exact (task_id, trajectory_id) pair (D-066).

        Args:
            verified_only: If True, return only records with success=True.
            task_id:        If provided, filter to this task.
            trajectory_id:  If provided, filter to this trajectory.
        """
        records = list(self._records)
        if verified_only:
            records = [r for r in records if r.success]
        if task_id is not None and trajectory_id is not None:
            # D-066: exact pair filtering
            pair = (task_id, trajectory_id)
            records = [r for r in records if r.trajectory_key == pair]
        elif task_id is not None:
            records = [r for r in records if r.task_id == task_id]
        elif trajectory_id is not None:
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

        Backward-compatible access (includes legacy records).  For the
        canonical procedural-SVD path, use ``canonical_delta_matrix()``.

        Returns an empty [0, D] tensor if no records pass the filter.
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
        delta_matrix() or canonical_delta_matrix(), not h_matrix().
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

    # -----------------------------------------------------------------------
    # Canonical procedural-SVD path (D-066: explicit provenance only)
    # -----------------------------------------------------------------------

    def canonical_delta_matrix(
        self,
        *,
        task_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> Tensor:
        """Canonical procedural-SVD input (D-066).

        Returns a [N, D] matrix of verified delta_h vectors from records
        that carry explicit task_id, trajectory_id, and structured
        success_evidence.  Legacy records (default IDs or bare boolean
        success) are excluded.

        When both task_id and trajectory_id are provided, filtering uses
        the exact (task_id, trajectory_id) pair.
        """
        records = [
            r for r in self._records
            if r.success and not _is_legacy_record(r)
        ]
        if task_id is not None and trajectory_id is not None:
            pair = (task_id, trajectory_id)
            records = [r for r in records if r.trajectory_key == pair]
        elif task_id is not None:
            records = [r for r in records if r.task_id == task_id]
        elif trajectory_id is not None:
            records = [r for r in records if r.trajectory_id == trajectory_id]

        if not records:
            if self._records:
                return torch.zeros(0, self._records[0].h_t.shape[0])
            return torch.zeros(0, 0)
        return torch.stack([r.delta_h for r in records], dim=0)

    def canonical_records(
        self,
        *,
        task_id: str | None = None,
        trajectory_id: str | None = None,
    ) -> list[TransitionRecord]:
        """Return verified records with explicit provenance (D-066).

        Excludes legacy records.  When both task_id and trajectory_id
        are provided, filtering uses the exact pair.
        """
        records = [
            r for r in self._records
            if r.success and not _is_legacy_record(r)
        ]
        if task_id is not None and trajectory_id is not None:
            pair = (task_id, trajectory_id)
            records = [r for r in records if r.trajectory_key == pair]
        elif task_id is not None:
            records = [r for r in records if r.task_id == task_id]
        elif trajectory_id is not None:
            records = [r for r in records if r.trajectory_id == trajectory_id]
        return records

    # -----------------------------------------------------------------------
    # Listing methods (D-066: pair-keyed)
    # -----------------------------------------------------------------------

    def trajectory_keys(self) -> list[tuple[str, str]]:
        """Return distinct (task_id, trajectory_id) pairs that have at least one record."""
        return list(dict.fromkeys(r.trajectory_key for r in self._records))

    def trajectories(self) -> list[str]:
        """Return distinct trajectory_ids that have at least one record.

        Note: this may contain duplicates across tasks.  Use
        ``trajectory_keys()`` for unique pair identification (D-066).
        """
        return list(dict.fromkeys(r.trajectory_id for r in self._records))

    def tasks(self) -> list[str]:
        """Return distinct task_ids that have at least one record."""
        return list(dict.fromkeys(r.task_id for r in self._records))

    # -----------------------------------------------------------------------
    # Counters
    # -----------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of records (including unverified and legacy)."""
        return len(self._records)

    @property
    def verified_count(self) -> int:
        """Number of verified (success=True) records."""
        return sum(1 for r in self._records if r.success)

    @property
    def canonical_count(self) -> int:
        """Number of verified records with explicit provenance (non-legacy)."""
        return sum(1 for r in self._records if r.success and not _is_legacy_record(r))

    @property
    def legacy_count(self) -> int:
        """Number of records tagged as legacy compatibility input."""
        return sum(1 for r in self._records if _is_legacy_record(r))
