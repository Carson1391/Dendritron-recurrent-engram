from __future__ import annotations

import unittest

from dendritron.capacity import (
    SparseCapacityLedger,
    required_compute_for_memory,
    solve_expert_geometry,
)
from dendritron.config import DendritronConfig


class CapacityLedgerTests(unittest.TestCase):
    def test_exact_fixed_split(self) -> None:
        ledger = SparseCapacityLedger(
            memory_parameters=1_000,
            compute_parameters=3_000,
            shared_core_parameters=500,
        )
        self.assertTrue(ledger.exact_25_75)
        self.assertEqual(ledger.memory_fraction, 0.25)
        self.assertEqual(ledger.compute_fraction, 0.75)
        ledger.assert_fixed_split()

    def test_mismatch_reports_exact_compute_gap(self) -> None:
        ledger = SparseCapacityLedger(
            memory_parameters=100,
            compute_parameters=240,
        )
        self.assertFalse(ledger.exact_25_75)
        self.assertEqual(ledger.compute_parameter_gap, 60)
        with self.assertRaises(ValueError):
            ledger.assert_fixed_split()

    def test_compute_is_three_times_memory(self) -> None:
        self.assertEqual(required_compute_for_memory(4_096), 12_288)

    def test_config_rejects_ratio_drift(self) -> None:
        with self.assertRaises(ValueError):
            DendritronConfig(vocab_size=259, memory_fraction=0.2)

    def test_tiny_runtime_has_exact_primary_capacity_split(self) -> None:
        try:
            from dendritron.config import tiny_smoke_config
            from dendritron.model import DendritronLM
        except ImportError:
            self.skipTest("PyTorch executes on the training host")
        ledger = DendritronLM(
            tiny_smoke_config(vocab_size=128)
        ).capacity_ledger()
        self.assertTrue(ledger.exact_25_75, ledger.as_dict())

    def test_expert_geometry_solver_recovers_exact_tiny_layout(self) -> None:
        plan = solve_expert_geometry(
            memory_parameters=592_896,
            fixed_compute_parameters=9_216,
            model_width=64,
            branch_count=1,
            minimum_experts=9,
            maximum_experts=9,
            minimum_hidden_width=512,
            maximum_hidden_width=512,
        )
        self.assertTrue(plan.exact_25_75)
        self.assertEqual(plan.expert_count, 9)
        self.assertEqual(plan.expert_hidden_width, 512)


if __name__ == "__main__":
    unittest.main()
