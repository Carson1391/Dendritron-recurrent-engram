from __future__ import annotations

import copy
import unittest

from stage3_jtd.corrected_stage2_contract import (
    validate_corrected_stage2_manifest,
)


def valid_manifest() -> dict:
    return {
        "completed": True,
        "states": {"hidden_state_layers": [8, 24]},
        "addressing": {"punctuation_boundary_aware": True},
        "tables": {
            "bigrams": {
                "rows": 500_000,
                "tensor_keys": ["layer08", "layer24"],
            },
            "trigrams": {
                "rows": 500_000,
                "tensor_keys": ["layer08", "layer24"],
            },
        },
        "sharding": {
            "tensor_keys": ["layer08", "layer24"],
            "both_layer_views_share_each_engram_row": True,
        },
        "migration": {
            "reusable_rows_copied": 809_775,
            "new_rows_extracted": 190_225,
            "retired_rows": 190_225,
            "total_corrected_rows": 1_000_000,
        },
    }


class CorrectedStage2ContractTests(unittest.TestCase):
    def test_accepts_reviewed_completed_bank(self) -> None:
        record = validate_corrected_stage2_manifest(valid_manifest())
        self.assertEqual(record["rows"]["bigrams"], 500_000)
        self.assertEqual(record["hidden_state_layers"], [8, 24])

    def test_rejects_original_punctuation_crossing_bank(self) -> None:
        manifest = valid_manifest()
        manifest["addressing"].pop("punctuation_boundary_aware")
        with self.assertRaisesRegex(ValueError, "punctuation-boundary"):
            validate_corrected_stage2_manifest(manifest)

    def test_rejects_changed_migration_totals(self) -> None:
        manifest = copy.deepcopy(valid_manifest())
        manifest["migration"]["new_rows_extracted"] -= 1
        with self.assertRaisesRegex(ValueError, "migration totals differ"):
            validate_corrected_stage2_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
