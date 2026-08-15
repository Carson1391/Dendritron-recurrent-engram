from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from stage2_delta.build_migration_plan import (
    build_migration_plan,
    extraction_positions,
    group_reusable_rows,
    json_fingerprint,
    load_plan_rows,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_old_keys(path: Path, rows: list[str], order: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row_index, text in enumerate(rows):
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "frequency": 100 - row_index,
                        "n": order,
                        "source_rank": row_index + 1,
                        "row_index": row_index,
                        "donor_token_count": order + 1,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_inventory(path: Path, rows: list[str], order: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rank, text in enumerate(rows, start=1):
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "frequency": 200 - rank,
                        "n": order,
                        "rank": rank,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class Stage2DeltaMigrationPlanTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, Path, dict]:
        old = root / "old-stage2"
        corrected = root / "corrected-stage1"
        output = root / "corrected-stage2"
        audit_path = root / "audit.json"

        old_bigrams = [
            "alpha beta",
            "cross punctuation",
            "rank mover",
            "Schrödinger equation",
        ]
        new_bigrams = [
            "rank mover",
            "alpha beta",
            "new clean",
            "Schrödinger equation",
        ]
        old_trigrams = [
            "one two three",
            "paper this result",
            "quantum field theory",
            "stable exact phrase",
        ]
        new_trigrams = [
            "stable exact phrase",
            "new scientific phrase",
            "quantum field theory",
            "another clean phrase",
        ]
        write_old_keys(old / "bigrams/keys.jsonl", old_bigrams, 2)
        write_old_keys(old / "trigrams/keys.jsonl", old_trigrams, 3)
        write_inventory(corrected / "top_bigrams.jsonl", new_bigrams, 2)
        write_inventory(corrected / "top_trigrams.jsonl", new_trigrams, 3)
        write_json(
            old / "manifest.json",
            {
                "completed": True,
                "model": {"resolved_revision": "fixed-commit"},
                "states": {
                    "hidden_state_layers": [8, 24],
                    "hidden_size": 2048,
                },
                "sharding": {"shard_size": 2},
            },
        )
        audit = {
            "completed": True,
            "orders": {
                "bigrams": {
                    "old_rows": 4,
                    "new_rows": 4,
                    "reusable_rows": 3,
                    "new_rows_to_extract": 1,
                    "old_rows_to_retire": 1,
                },
                "trigrams": {
                    "old_rows": 4,
                    "new_rows": 4,
                    "reusable_rows": 2,
                    "new_rows_to_extract": 2,
                    "old_rows_to_retire": 2,
                },
            },
        }
        write_json(audit_path, audit)
        approved = {
            "bigrams": {
                "old_rows": 4,
                "corrected_rows": 4,
                "reusable_rows": 3,
                "new_rows_to_extract": 1,
                "retired_rows": 1,
            },
            "trigrams": {
                "old_rows": 4,
                "corrected_rows": 4,
                "reusable_rows": 2,
                "new_rows_to_extract": 2,
                "retired_rows": 2,
            },
        }
        return old, corrected, audit_path, output, approved

    def test_exact_key_plan_preserves_vectors_across_rank_movement(self) -> None:
        with TemporaryDirectory() as temporary:
            old, corrected, audit, output, approved = self._fixture(
                Path(temporary)
            )
            plan = build_migration_plan(
                old_stage2_root=old,
                corrected_stage1_root=corrected,
                audit_path=audit,
                output_root=output,
                destination_shard_size=2,
                expected_rows_per_order=4,
                approved_counts=approved,
            )

            self.assertEqual(plan["totals"]["corrected_rows"], 8)
            self.assertEqual(plan["totals"]["reusable_rows"], 5)
            self.assertEqual(plan["totals"]["new_rows_to_extract"], 3)
            unsigned = dict(plan)
            fingerprint = unsigned.pop("fingerprint")
            self.assertEqual(fingerprint, json_fingerprint(unsigned))

            first_path = Path(
                plan["tables"]["bigrams"]["plan_files"][0]["path"]
            )
            rows = load_plan_rows(first_path, expected_order=2)
            self.assertEqual([row["text"] for row in rows], ["rank mover", "alpha beta"])
            self.assertEqual(
                [row["source_stage2_row_index"] for row in rows],
                [2, 0],
            )
            self.assertEqual(
                group_reusable_rows(rows),
                {1: [(0, 0)], 0: [(1, 0)]},
            )
            self.assertEqual(extraction_positions(rows), [])

            # Simulate the exact scatter performed for each donor layer.  The
            # destination order follows corrected rank while vector identity
            # follows the old phrase row across source-shard boundaries.
            old_layer = {
                0: np.asarray([[10.0, 11.0], [20.0, 21.0]]),
                1: np.asarray([[30.0, 31.0], [40.0, 41.0]]),
            }
            destination = np.empty((2, 2))
            for source_shard, pairs in group_reusable_rows(rows).items():
                for destination_local, source_local in pairs:
                    destination[destination_local] = old_layer[source_shard][
                        source_local
                    ]
            np.testing.assert_array_equal(
                destination,
                np.asarray([[30.0, 31.0], [10.0, 11.0]]),
            )

            second_path = Path(
                plan["tables"]["bigrams"]["plan_files"][1]["path"]
            )
            second_rows = load_plan_rows(second_path, expected_order=2)
            self.assertEqual(extraction_positions(second_rows), [0])
            self.assertEqual(
                second_rows[1]["text"],
                "Schrödinger equation",
            )
            self.assertEqual(
                second_rows[1]["source_stage2_row_index"],
                3,
            )

    def test_audit_mismatch_stops_before_materialization(self) -> None:
        with TemporaryDirectory() as temporary:
            old, corrected, audit_path, output, approved = self._fixture(
                Path(temporary)
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["orders"]["bigrams"]["reusable_rows"] = 2
            write_json(audit_path, audit)

            with self.assertRaisesRegex(ValueError, "audit/plan mismatch"):
                build_migration_plan(
                    old_stage2_root=old,
                    corrected_stage1_root=corrected,
                    audit_path=audit_path,
                    output_root=output,
                    destination_shard_size=2,
                    expected_rows_per_order=4,
                    approved_counts=approved,
                )

    def test_exact_utf8_identity_does_not_fold_accents(self) -> None:
        with TemporaryDirectory() as temporary:
            old, corrected, audit_path, output, approved = self._fixture(
                Path(temporary)
            )
            inventory_path = corrected / "top_bigrams.jsonl"
            rows = [json.loads(line) for line in inventory_path.read_text(encoding="utf-8").splitlines()]
            rows[-1]["text"] = "Schrodinger equation"
            inventory_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["orders"]["bigrams"].update(
                {
                    "reusable_rows": 2,
                    "new_rows_to_extract": 2,
                    "old_rows_to_retire": 2,
                }
            )
            write_json(audit_path, audit)
            approved["bigrams"].update(
                {
                    "reusable_rows": 2,
                    "new_rows_to_extract": 2,
                    "retired_rows": 2,
                }
            )
            plan = build_migration_plan(
                old_stage2_root=old,
                corrected_stage1_root=corrected,
                audit_path=audit_path,
                output_root=output,
                destination_shard_size=2,
                expected_rows_per_order=4,
                approved_counts=approved,
            )
            second_path = Path(
                plan["tables"]["bigrams"]["plan_files"][1]["path"]
            )
            second_rows = load_plan_rows(second_path, expected_order=2)
            self.assertEqual(extraction_positions(second_rows), [0, 1])


if __name__ == "__main__":
    unittest.main()
