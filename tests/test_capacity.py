from __future__ import annotations

import unittest

from dendritron.capacity import (
    SparseCapacityLedger,
    required_compute_for_memory,
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


if __name__ == "__main__":
    unittest.main()
