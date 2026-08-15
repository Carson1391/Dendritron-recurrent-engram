from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from stage3_jtd.compare_punctuation_inventories import compare


def write_rows(path: Path, rows: list[tuple[str, int]], order: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rank, (text, frequency) in enumerate(rows, start=1):
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "frequency": frequency,
                        "n": order,
                        "rank": rank,
                    }
                )
                + "\n"
            )


class PunctuationInventoryAuditTests(unittest.TestCase):
    def test_reports_reusable_and_replacement_rows(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old"
            new = root / "new"
            output = root / "audit"
            write_rows(
                old / "bigrams" / "keys.jsonl",
                [("tree bark", 10), ("paper this", 7)],
                2,
            )
            write_rows(
                old / "trigrams" / "keys.jsonl",
                [("rough tree bark", 9), ("result this paper", 6)],
                3,
            )
            write_rows(
                new / "top_bigrams.jsonl",
                [("tree bark", 10), ("neural network", 8)],
                2,
            )
            write_rows(
                new / "top_trigrams.jsonl",
                [("rough tree bark", 9), ("quantum field theory", 7)],
                3,
            )

            result = compare(
                Namespace(old_stage2=old, new_stage1=new, output=output)
            )

            self.assertEqual(result["totals"]["corrected_rows"], 4)
            self.assertEqual(result["totals"]["reusable_frozen_rows"], 2)
            self.assertEqual(result["totals"]["new_donor_rows_to_extract"], 2)
            self.assertEqual(result["totals"]["old_donor_rows_to_retire"], 2)
            self.assertEqual(
                result["totals"]["distinct_words_in_new_rows"],
                5,
            )
            added = [
                json.loads(line)["text"]
                for line in (output / "bigrams_new_rows.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(added, ["neural network"])

    def test_rejects_discontinuous_rank_contract(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old"
            new = root / "new"
            output = root / "audit"
            write_rows(old / "bigrams" / "keys.jsonl", [("a b", 2)], 2)
            write_rows(old / "trigrams" / "keys.jsonl", [("a b c", 2)], 3)
            write_rows(new / "top_bigrams.jsonl", [("a b", 2)], 2)
            write_rows(new / "top_trigrams.jsonl", [("a b c", 2)], 3)
            row = json.loads((new / "top_bigrams.jsonl").read_text())
            row["rank"] = 2
            (new / "top_bigrams.jsonl").write_text(json.dumps(row) + "\n")

            with self.assertRaises(ValueError):
                compare(Namespace(old_stage2=old, new_stage1=new, output=output))


if __name__ == "__main__":
    unittest.main()
