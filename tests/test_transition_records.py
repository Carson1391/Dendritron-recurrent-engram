"""Procedural-delta tests for D-063.

D-063: Procedural-SVD input consists of verified transition records with
delta_h_t = h_{t+1} - h_t, skill/block/round identifiers, success evidence,
and ordering metadata.

Acceptance:
  - stalled-state zero deltas (h_{t+1} == h_t => delta_h == 0)
  - translation invariance under h_t -> h_t + c
  - verified-trajectory filtering (only success=True records pass)
  - entry-point contract accepts transition records
"""

from __future__ import annotations

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class D063TransitionRecordTests(unittest.TestCase):
    """D-063: Verified transition records for procedural SVD input."""

    def setUp(self) -> None:
        import torch

        torch.manual_seed(123)

    def test_delta_is_h_next_minus_h_t(self) -> None:
        """delta_h_t must equal h_{t+1} - h_t exactly."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h_t = torch.randn(64)
        h_next = torch.randn(64)

        record = builder.add_visit(h_t, h_next, block_index=0, round_index=0)

        self.assertTrue(
            torch.allclose(record.delta_h, h_next - h_t, atol=1e-7),
            "delta_h != h_next - h_t",
        )

    def test_stalled_state_produces_zero_delta(self) -> None:
        """When h_{t+1} == h_t, delta_h must be exactly zero."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(64)

        record = builder.add_visit(h, h.clone(), block_index=0, round_index=0)

        self.assertTrue(
            torch.allclose(record.delta_h, torch.zeros_like(h), atol=0.0),
            "Stalled state did not produce exactly zero delta",
        )

    def test_translation_invariance(self) -> None:
        """delta_h must be invariant under h_t -> h_t + c.

        If both h_t and h_{t+1} are shifted by the same constant c,
        delta_h = (h_{t+1} + c) - (h_t + c) = h_{t+1} - h_t.
        """
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h_t = torch.randn(64)
        h_next = torch.randn(64)
        c = torch.randn(64) * 10.0  # arbitrary constant shift

        record_base = builder.add_visit(h_t, h_next, block_index=0, round_index=0)
        record_shifted = builder.add_visit(
            h_t + c, h_next + c, block_index=0, round_index=0
        )

        self.assertTrue(
            torch.allclose(record_base.delta_h, record_shifted.delta_h, atol=1e-6),
            "Translation invariance violated: delta_h changed under h_t -> h_t + c",
        )

    def test_verified_trajectory_filtering(self) -> None:
        """Only success=True records must appear in verified output."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()

        # Add 5 records: 3 successful, 2 failed
        for i in range(5):
            h_t = torch.randn(32)
            h_next = torch.randn(32)
            success = (i % 2 == 0)  # 0, 2, 4 are successful
            builder.add_visit(
                h_t, h_next,
                skill_ids=(i,),
                block_index=i % 2,
                round_index=0,
                success=success,
            )

        self.assertEqual(builder.count, 5)
        self.assertEqual(builder.verified_count, 3)

        verified = builder.build(verified_only=True)
        self.assertEqual(len(verified), 3)
        for r in verified:
            self.assertTrue(r.success)

        all_records = builder.build(verified_only=False)
        self.assertEqual(len(all_records), 5)

    def test_delta_matrix_shape_and_content(self) -> None:
        """delta_matrix must return [N, D] with correct deltas."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        dim = 48
        for i in range(6):
            h_t = torch.randn(dim)
            h_next = torch.randn(dim)
            builder.add_visit(h_t, h_next, block_index=0, round_index=0)

        matrix = builder.delta_matrix(verified_only=True)
        self.assertEqual(matrix.shape, (6, dim))

        # Verify each row matches the corresponding delta
        records = builder.build(verified_only=True)
        for i, r in enumerate(records):
            self.assertTrue(
                torch.allclose(matrix[i], r.delta_h, atol=1e-7),
                f"delta_matrix row {i} does not match record delta_h",
            )

    def test_delta_matrix_empty_when_no_verified_records(self) -> None:
        """delta_matrix must return [0, D] when all records are unverified."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h_t = torch.randn(32)
        h_next = torch.randn(32)
        builder.add_visit(h_t, h_next, success=False)

        matrix = builder.delta_matrix(verified_only=True)
        self.assertEqual(matrix.shape, (0, 32))

    def test_skill_ids_block_round_preserved(self) -> None:
        """Records must carry skill_ids, block_index, and round_index."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h_t = torch.randn(16)
        h_next = torch.randn(16)

        record = builder.add_visit(
            h_t, h_next,
            skill_ids=(2, 5, 7),
            block_index=1,
            round_index=3,
            success=True,
        )

        self.assertEqual(record.skill_ids, (2, 5, 7))
        self.assertEqual(record.block_index, 1)
        self.assertEqual(record.round_index, 3)
        self.assertTrue(record.success)

    def test_ordering_metadata_is_monotonic(self) -> None:
        """Ordering field must be monotonically increasing."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        for i in range(5):
            h = torch.randn(8)
            builder.add_visit(h, h + 0.1, block_index=0, round_index=0)

        records = builder.build(verified_only=False)
        orderings = [r.ordering for r in records]
        self.assertEqual(orderings, [0, 1, 2, 3, 4])

    def test_entry_point_contract_accepts_transition_records(self) -> None:
        """The builder must accept transition records and return them as a list.

        This verifies the entry-point contract: the SVD consumer can call
        build() and delta_matrix() and receive valid data structures.
        """
        import torch

        from dendritron.transition_records import (
            TransitionRecord,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        dim = 32
        for i in range(4):
            builder.add_visit(
                torch.randn(dim),
                torch.randn(dim),
                skill_ids=(i,),
                block_index=i % 2,
                round_index=i // 2,
                success=True,
            )

        records = builder.build(verified_only=True)
        self.assertTrue(all(isinstance(r, TransitionRecord) for r in records))
        self.assertEqual(len(records), 4)

        # delta_matrix is a valid [N, D] tensor for SVD
        matrix = builder.delta_matrix()
        self.assertEqual(matrix.ndim, 2)
        self.assertEqual(matrix.shape[0], 4)
        self.assertEqual(matrix.shape[1], dim)
        self.assertTrue(torch.isfinite(matrix).all())

    def test_h_matrix_returns_pre_transition_states(self) -> None:
        """h_matrix must return [N, D] pre-transition h_t vectors."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        dim = 24
        h_states = []
        for i in range(3):
            h_t = torch.randn(dim)
            h_next = torch.randn(dim)
            h_states.append(h_t)
            builder.add_visit(h_t, h_next, block_index=0, round_index=0)

        h_mat = builder.h_matrix(verified_only=True)
        self.assertEqual(h_mat.shape, (3, dim))
        for i in range(3):
            self.assertTrue(
                torch.allclose(h_mat[i], h_states[i], atol=1e-7),
                f"h_matrix row {i} does not match original h_t",
            )

    def test_record_validation_rejects_mismatched_shapes(self) -> None:
        """add_visit must reject h_t and h_next with different shapes."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h_t = torch.randn(32)
        h_next = torch.randn(64)

        with self.assertRaises(ValueError):
            builder.add_visit(h_t, h_next)

    def test_record_validation_rejects_bad_block_index(self) -> None:
        """add_visit must reject block_index outside {0, 1}."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(16)

        with self.assertRaises(ValueError):
            builder.add_visit(h, h, block_index=2)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class D065RecordProvenanceTests(unittest.TestCase):
    """D-065: Task/trajectory identity, structured success evidence,
    per-trajectory ordering, optional skill_ids, detached clones,
    and delta_matrix as canonical procedural-SVD input."""

    def setUp(self) -> None:
        import torch

        torch.manual_seed(77)

    def test_task_id_and_trajectory_id_preserved(self) -> None:
        """Records must carry task_id and trajectory_id."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(32)
        record = builder.add_visit(
            h, h + 0.1,
            task_id="arithmetic",
            trajectory_id="traj-001",
        )

        self.assertEqual(record.task_id, "arithmetic")
        self.assertEqual(record.trajectory_id, "traj-001")

    def test_structured_success_evidence_stores_reason_and_metrics(self) -> None:
        """success_evidence must carry verdict, reason, and metrics."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(16)
        evidence = SuccessEvidence(
            verdict=True,
            reason="correct sum: 2+2=4",
            metrics={"loss": 0.01, "rank_margin": 3.5},
        )
        record = builder.add_visit(
            h, h + 0.1,
            success_evidence=evidence,
        )

        self.assertEqual(record.success_evidence.verdict, True)
        self.assertEqual(record.success_evidence.reason, "correct sum: 2+2=4")
        self.assertAlmostEqual(record.success_evidence.metrics["loss"], 0.01)
        self.assertAlmostEqual(record.success_evidence.metrics["rank_margin"], 3.5)

    def test_success_derived_from_evidence_not_stored_independently(self) -> None:
        """success must be a read-only property derived from success_evidence."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(16)

        # False verdict
        record_fail = builder.add_visit(
            h, h + 0.1,
            success_evidence=SuccessEvidence(verdict=False, reason="wrong answer"),
        )
        self.assertFalse(record_fail.success)
        self.assertFalse(record_fail.success_evidence.success)

        # True verdict
        record_ok = builder.add_visit(
            h, h + 0.2,
            success_evidence=SuccessEvidence(verdict=True, reason="correct"),
        )
        self.assertTrue(record_ok.success)
        self.assertTrue(record_ok.success_evidence.success)

    def test_ordering_scoped_per_trajectory(self) -> None:
        """Ordering must reset to 0 for each new trajectory_id."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(8)

        # Trajectory A: 3 visits -> orderings 0, 1, 2
        for _ in range(3):
            builder.add_visit(h, h + 0.1, trajectory_id="traj-A")

        # Trajectory B: 2 visits -> orderings 0, 1
        for _ in range(2):
            builder.add_visit(h, h + 0.1, trajectory_id="traj-B")

        # Trajectory A again: ordering continues from 3
        builder.add_visit(h, h + 0.1, trajectory_id="traj-A")

        records_a = builder.build(verified_only=False, trajectory_id="traj-A")
        orderings_a = [r.ordering for r in records_a]
        self.assertEqual(orderings_a, [0, 1, 2, 3])

        records_b = builder.build(verified_only=False, trajectory_id="traj-B")
        orderings_b = [r.ordering for r in records_b]
        self.assertEqual(orderings_b, [0, 1])

    def test_empty_skill_ids_allowed_during_initial_discovery(self) -> None:
        """skill_ids may be empty — SVD discovers slots after capture."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(32)

        record = builder.add_visit(h, h + 0.1, skill_ids=())
        self.assertEqual(record.skill_ids, ())

        # Also test default (no skill_ids kwarg)
        record2 = builder.add_visit(h, h + 0.2)
        self.assertEqual(record2.skill_ids, ())

    def test_tensors_are_detached_from_computation_graph(self) -> None:
        """All tensor snapshots must be detached — no grad connection."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h_t = torch.randn(32, requires_grad=True)
        h_next = torch.randn(32, requires_grad=True)

        record = builder.add_visit(h_t, h_next)

        self.assertFalse(record.h_t.requires_grad)
        self.assertFalse(record.h_next.requires_grad)
        self.assertFalse(record.delta_h.requires_grad)

    def test_detached_clones_preserve_exact_values(self) -> None:
        """Detached clones must preserve the exact numerical values."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h_t = torch.randn(64)
        h_next = torch.randn(64)

        record = builder.add_visit(h_t, h_next)

        self.assertTrue(torch.allclose(record.h_t, h_t, atol=0.0))
        self.assertTrue(torch.allclose(record.h_next, h_next, atol=0.0))
        self.assertTrue(torch.allclose(record.delta_h, h_next - h_t, atol=1e-7))

    def test_detached_clones_are_independent_copies(self) -> None:
        """Modifying the source tensor after capture must not affect the record."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h_t = torch.randn(32)
        h_next = torch.randn(32)

        record = builder.add_visit(h_t, h_next)

        # Mutate the originals
        h_t.add_(100.0)
        h_next.add_(100.0)

        # Record must be unaffected
        self.assertFalse(torch.allclose(record.h_t, h_t))
        self.assertFalse(torch.allclose(record.h_next, h_next))

    def test_delta_matrix_is_canonical_svd_input(self) -> None:
        """delta_matrix must be the canonical procedural-SVD input."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        dim = 48
        for i in range(5):
            evidence = SuccessEvidence(
                verdict=(i < 3),
                reason=f"step {i}",
                metrics={"step": float(i)},
            )
            builder.add_visit(
                torch.randn(dim),
                torch.randn(dim),
                task_id="task-1",
                trajectory_id="traj-1",
                success_evidence=evidence,
            )

        # Only 3 verified records
        delta_mat = builder.delta_matrix(verified_only=True)
        self.assertEqual(delta_mat.shape, (3, dim))
        self.assertTrue(torch.isfinite(delta_mat).all())

        # Unverified records excluded
        all_delta = builder.delta_matrix(verified_only=False)
        self.assertEqual(all_delta.shape, (5, dim))

    def test_h_matrix_remains_diagnostic_view(self) -> None:
        """h_matrix must remain available as a diagnostic view."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        dim = 32
        for _ in range(3):
            builder.add_visit(
                torch.randn(dim),
                torch.randn(dim),
                task_id="t",
                trajectory_id="tr",
            )

        h_mat = builder.h_matrix(verified_only=True)
        self.assertEqual(h_mat.shape, (3, dim))

    def test_filter_by_task_id(self) -> None:
        """build() and delta_matrix() must filter by task_id."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(16)

        for i in range(4):
            builder.add_visit(
                h, h + 0.1,
                task_id="task-A" if i < 2 else "task-B",
                trajectory_id=f"traj-{i}",
            )

        task_a = builder.build(task_id="task-A")
        self.assertEqual(len(task_a), 2)
        self.assertTrue(all(r.task_id == "task-A" for r in task_a))

        task_b = builder.build(task_id="task-B")
        self.assertEqual(len(task_b), 2)
        self.assertTrue(all(r.task_id == "task-B" for r in task_b))

        delta_a = builder.delta_matrix(task_id="task-A")
        self.assertEqual(delta_a.shape, (2, 16))

    def test_filter_by_trajectory_id(self) -> None:
        """build() and delta_matrix() must filter by trajectory_id."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(16)

        for i in range(3):
            builder.add_visit(
                h, h + 0.1,
                task_id="task-X",
                trajectory_id=f"traj-{i}",
            )

        traj_0 = builder.build(trajectory_id="traj-0")
        self.assertEqual(len(traj_0), 1)
        self.assertEqual(traj_0[0].trajectory_id, "traj-0")

    def test_trajectories_and_tasks_listing(self) -> None:
        """trajectories() and tasks() must list distinct IDs."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(8)

        builder.add_visit(h, h, task_id="t1", trajectory_id="tr1")
        builder.add_visit(h, h, task_id="t1", trajectory_id="tr2")
        builder.add_visit(h, h, task_id="t2", trajectory_id="tr3")

        self.assertEqual(builder.tasks(), ["t1", "t2"])
        self.assertEqual(builder.trajectories(), ["tr1", "tr2", "tr3"])

    def test_backward_compatible_success_param(self) -> None:
        """The success= bool parameter must still work for backward compatibility."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(16)

        record_ok = builder.add_visit(h, h + 0.1, success=True)
        record_fail = builder.add_visit(h, h + 0.2, success=False)

        self.assertTrue(record_ok.success)
        self.assertFalse(record_fail.success)
        self.assertTrue(record_ok.success_evidence.verdict)
        self.assertFalse(record_fail.success_evidence.verdict)

    def test_empty_task_id_rejected(self) -> None:
        """task_id must be a non-empty string."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(16)

        with self.assertRaises(ValueError):
            builder.add_visit(h, h + 0.1, task_id="")

    def test_empty_trajectory_id_rejected(self) -> None:
        """trajectory_id must be a non-empty string."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(16)

        with self.assertRaises(ValueError):
            builder.add_visit(h, h + 0.1, trajectory_id="")

    def test_negative_skill_id_rejected(self) -> None:
        """Negative skill_ids must be rejected."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(16)

        with self.assertRaises(ValueError):
            builder.add_visit(h, h + 0.1, skill_ids=(-1,))

    def test_success_evidence_to_dict(self) -> None:
        """SuccessEvidence.to_dict must serialize the evidence payload."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        evidence = SuccessEvidence(
            verdict=True,
            reason="test passed",
            metrics={"loss": 0.05},
        )
        d = evidence.to_dict()
        self.assertEqual(d["verdict"], True)
        self.assertEqual(d["reason"], "test passed")
        self.assertAlmostEqual(d["metrics"]["loss"], 0.05)

    def test_multi_trajectory_delta_matrix(self) -> None:
        """delta_matrix must correctly stack across multiple trajectories."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        dim = 24

        # Trajectory 1: 2 verified + 1 failed
        builder.add_visit(torch.randn(dim), torch.randn(dim),
                          task_id="t", trajectory_id="tr1",
                          success_evidence=SuccessEvidence(True, "ok"))
        builder.add_visit(torch.randn(dim), torch.randn(dim),
                          task_id="t", trajectory_id="tr1",
                          success_evidence=SuccessEvidence(False, "bad"))
        builder.add_visit(torch.randn(dim), torch.randn(dim),
                          task_id="t", trajectory_id="tr1",
                          success_evidence=SuccessEvidence(True, "ok"))

        # Trajectory 2: 3 verified
        for _ in range(3):
            builder.add_visit(torch.randn(dim), torch.randn(dim),
                              task_id="t", trajectory_id="tr2",
                              success_evidence=SuccessEvidence(True, "ok"))

        # All verified: 2 + 3 = 5
        all_delta = builder.delta_matrix(verified_only=True)
        self.assertEqual(all_delta.shape, (5, dim))

        # Only trajectory 1: 2 verified
        tr1_delta = builder.delta_matrix(verified_only=True, trajectory_id="tr1")
        self.assertEqual(tr1_delta.shape, (2, dim))

        # Only trajectory 2: 3 verified
        tr2_delta = builder.delta_matrix(verified_only=True, trajectory_id="tr2")
        self.assertEqual(tr2_delta.shape, (3, dim))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class D066PairKeyProvenanceTests(unittest.TestCase):
    """D-066: Trajectory identity by (task_id, trajectory_id) pair.

    Two tasks using the same trajectory_id (e.g. traj-001) must get
    independent ordering counters, independent filtering, and independent
    listing.  Legacy compatibility inputs are tagged and excluded from
    the canonical procedural-SVD path.
    """

    def setUp(self) -> None:
        import torch

        torch.manual_seed(256)

    def test_two_tasks_same_traj_id_each_begin_at_ordering_zero(self) -> None:
        """task-A/traj-001 and task-B/traj-001 must each start at ordering 0."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(16)
        ev = SuccessEvidence(verdict=True, reason="ok")

        # task-A / traj-001: 3 visits
        for _ in range(3):
            builder.add_visit(
                h, h + 0.1, task_id="task-A", trajectory_id="traj-001",
                success_evidence=ev,
            )

        # task-B / traj-001: 2 visits
        for _ in range(2):
            builder.add_visit(
                h, h + 0.1, task_id="task-B", trajectory_id="traj-001",
                success_evidence=ev,
            )

        records_a = builder.build(task_id="task-A", trajectory_id="traj-001")
        records_b = builder.build(task_id="task-B", trajectory_id="traj-001")

        orderings_a = [r.ordering for r in records_a]
        orderings_b = [r.ordering for r in records_b]

        self.assertEqual(orderings_a, [0, 1, 2])
        self.assertEqual(orderings_b, [0, 1])

    def test_pair_key_filtering_is_exact(self) -> None:
        """build() with both task_id and trajectory_id must filter by the pair."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)
        ev = SuccessEvidence(verdict=True, reason="ok")

        builder.add_visit(h, h, task_id="A", trajectory_id="tr1", success_evidence=ev)
        builder.add_visit(h, h, task_id="A", trajectory_id="tr2", success_evidence=ev)
        builder.add_visit(h, h, task_id="B", trajectory_id="tr1", success_evidence=ev)
        builder.add_visit(h, h, task_id="B", trajectory_id="tr2", success_evidence=ev)

        # Exact pair (A, tr1) -> 1 record
        a_tr1 = builder.build(task_id="A", trajectory_id="tr1")
        self.assertEqual(len(a_tr1), 1)
        self.assertEqual(a_tr1[0].task_id, "A")
        self.assertEqual(a_tr1[0].trajectory_id, "tr1")

        # Exact pair (B, tr1) -> 1 record (different from A/tr1)
        b_tr1 = builder.build(task_id="B", trajectory_id="tr1")
        self.assertEqual(len(b_tr1), 1)
        self.assertEqual(b_tr1[0].task_id, "B")

        # Just task_id A -> 2 records
        a_all = builder.build(task_id="A")
        self.assertEqual(len(a_all), 2)

        # Just trajectory_id tr1 -> 2 records (from A and B)
        tr1_all = builder.build(trajectory_id="tr1")
        self.assertEqual(len(tr1_all), 2)

    def test_trajectory_key_property(self) -> None:
        """TransitionRecord.trajectory_key must return (task_id, trajectory_id)."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)
        record = builder.add_visit(
            h, h + 0.1,
            task_id="task-X",
            trajectory_id="traj-Y",
            success_evidence=SuccessEvidence(verdict=True, reason="ok"),
        )

        self.assertEqual(record.trajectory_key, ("task-X", "traj-Y"))

    def test_trajectory_keys_listing_returns_pairs(self) -> None:
        """trajectory_keys() must return distinct (task_id, trajectory_id) pairs."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)
        ev = SuccessEvidence(verdict=True, reason="ok")

        builder.add_visit(h, h, task_id="A", trajectory_id="tr1", success_evidence=ev)
        builder.add_visit(h, h, task_id="A", trajectory_id="tr2", success_evidence=ev)
        builder.add_visit(h, h, task_id="B", trajectory_id="tr1", success_evidence=ev)

        keys = builder.trajectory_keys()
        self.assertEqual(keys, [("A", "tr1"), ("A", "tr2"), ("B", "tr1")])

    def test_trajectories_listing_may_contain_duplicates_across_tasks(self) -> None:
        """trajectories() returns trajectory_ids only — may duplicate across tasks."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)
        ev = SuccessEvidence(verdict=True, reason="ok")

        builder.add_visit(h, h, task_id="A", trajectory_id="tr1", success_evidence=ev)
        builder.add_visit(h, h, task_id="B", trajectory_id="tr1", success_evidence=ev)

        # trajectories() returns ["tr1"] — only one unique trajectory_id
        self.assertEqual(builder.trajectories(), ["tr1"])
        # But trajectory_keys() returns two distinct pairs
        self.assertEqual(builder.trajectory_keys(), [("A", "tr1"), ("B", "tr1")])

    def test_legacy_records_excluded_from_canonical_delta_matrix(self) -> None:
        """canonical_delta_matrix must exclude legacy compatibility records."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(32)

        # Legacy: bare success= without explicit task/trajectory/evidence
        builder.add_visit(h, h + 0.1, success=True)
        builder.add_visit(h, h + 0.2, success=False)

        # Canonical: explicit task, trajectory, structured evidence
        builder.add_visit(
            h, h + 0.3,
            task_id="task-1", trajectory_id="traj-1",
            success_evidence=SuccessEvidence(verdict=True, reason="ok"),
        )
        builder.add_visit(
            h, h + 0.4,
            task_id="task-1", trajectory_id="traj-1",
            success_evidence=SuccessEvidence(verdict=False, reason="bad"),
        )
        builder.add_visit(
            h, h + 0.5,
            task_id="task-1", trajectory_id="traj-1",
            success_evidence=SuccessEvidence(verdict=True, reason="ok"),
        )

        # Canonical: only 2 verified non-legacy records
        canon = builder.canonical_delta_matrix()
        self.assertEqual(canon.shape, (2, 32))

        # Backward-compatible delta_matrix includes legacy
        all_verified = builder.delta_matrix(verified_only=True)
        # 1 legacy success + 2 canonical success = 3
        self.assertEqual(all_verified.shape, (3, 32))

    def test_canonical_delta_matrix_filters_by_pair(self) -> None:
        """canonical_delta_matrix with task_id + trajectory_id filters by pair."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(16)
        ev = SuccessEvidence(verdict=True, reason="ok")

        for _ in range(2):
            builder.add_visit(h, h + 0.1, task_id="A", trajectory_id="tr1", success_evidence=ev)
        for _ in range(3):
            builder.add_visit(h, h + 0.1, task_id="B", trajectory_id="tr1", success_evidence=ev)

        # Both tasks use "tr1" — pair filter must distinguish them
        a_tr1 = builder.canonical_delta_matrix(task_id="A", trajectory_id="tr1")
        self.assertEqual(a_tr1.shape, (2, 16))

        b_tr1 = builder.canonical_delta_matrix(task_id="B", trajectory_id="tr1")
        self.assertEqual(b_tr1.shape, (3, 16))

    def test_canonical_records_excludes_legacy(self) -> None:
        """canonical_records must return only non-legacy verified records."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)

        builder.add_visit(h, h, success=True)  # legacy
        builder.add_visit(
            h, h,
            task_id="t", trajectory_id="tr",
            success_evidence=SuccessEvidence(verdict=True, reason="ok"),
        )  # canonical

        canon = builder.canonical_records()
        self.assertEqual(len(canon), 1)
        self.assertEqual(canon[0].task_id, "t")
        self.assertEqual(canon[0].trajectory_id, "tr")

    def test_legacy_count_property(self) -> None:
        """legacy_count must report the number of legacy records."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)

        builder.add_visit(h, h, success=True)  # legacy
        builder.add_visit(h, h, success=False)  # legacy
        builder.add_visit(
            h, h,
            task_id="t", trajectory_id="tr",
            success_evidence=SuccessEvidence(verdict=True, reason="ok"),
        )  # canonical

        self.assertEqual(builder.legacy_count, 2)
        self.assertEqual(builder.canonical_count, 1)
        self.assertEqual(builder.count, 3)

    def test_canonical_count_excludes_legacy_and_unverified(self) -> None:
        """canonical_count must count only verified non-legacy records."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)

        # Canonical but failed
        builder.add_visit(
            h, h,
            task_id="t", trajectory_id="tr",
            success_evidence=SuccessEvidence(verdict=False, reason="bad"),
        )
        # Canonical and verified
        builder.add_visit(
            h, h,
            task_id="t", trajectory_id="tr",
            success_evidence=SuccessEvidence(verdict=True, reason="ok"),
        )
        # Legacy and verified
        builder.add_visit(h, h, success=True)

        self.assertEqual(builder.canonical_count, 1)
        self.assertEqual(builder.verified_count, 2)
        self.assertEqual(builder.legacy_count, 1)

    def test_ordering_continues_correctly_after_interleaving(self) -> None:
        """Ordering per pair must continue correctly when tasks interleave."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)
        ev = SuccessEvidence(verdict=True, reason="ok")

        # Interleave A/tr1 and B/tr1
        builder.add_visit(h, h, task_id="A", trajectory_id="tr1", success_evidence=ev)
        builder.add_visit(h, h, task_id="B", trajectory_id="tr1", success_evidence=ev)
        builder.add_visit(h, h, task_id="A", trajectory_id="tr1", success_evidence=ev)
        builder.add_visit(h, h, task_id="B", trajectory_id="tr1", success_evidence=ev)
        builder.add_visit(h, h, task_id="A", trajectory_id="tr1", success_evidence=ev)

        a_records = builder.build(task_id="A", trajectory_id="tr1")
        b_records = builder.build(task_id="B", trajectory_id="tr1")

        self.assertEqual([r.ordering for r in a_records], [0, 1, 2])
        self.assertEqual([r.ordering for r in b_records], [0, 1])

    def test_default_add_visit_creates_legacy_record(self) -> None:
        """add_visit with no explicit task/trajectory/evidence creates legacy."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(8)

        record = builder.add_visit(h, h + 0.1)

        self.assertTrue(_is_legacy(record))

    def test_explicit_task_with_bare_success_is_legacy(self) -> None:
        """add_visit with explicit task_id but bare success= is still legacy."""
        import torch

        from dendritron.transition_records import TransitionRecordBuilder

        builder = TransitionRecordBuilder()
        h = torch.randn(8)

        record = builder.add_visit(
            h, h + 0.1,
            task_id="task-A", trajectory_id="traj-1",
            success=True,
        )

        # Legacy because success_evidence was not explicitly provided
        self.assertTrue(_is_legacy(record))

    def test_fully_explicit_record_is_not_legacy(self) -> None:
        """add_visit with explicit task, trajectory, and success_evidence is canonical."""
        import torch

        from dendritron.transition_records import (
            SuccessEvidence,
            TransitionRecordBuilder,
        )

        builder = TransitionRecordBuilder()
        h = torch.randn(8)

        record = builder.add_visit(
            h, h + 0.1,
            task_id="task-A", trajectory_id="traj-1",
            success_evidence=SuccessEvidence(verdict=True, reason="ok"),
        )

        self.assertFalse(_is_legacy(record))


def _is_legacy(record) -> bool:
    """Helper: check if a record is tagged as legacy."""
    from dendritron.transition_records import _is_legacy_record
    return _is_legacy_record(record)


if __name__ == "__main__":
    unittest.main()
