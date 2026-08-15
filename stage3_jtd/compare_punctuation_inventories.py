"""Compare the completed Engram keys with a punctuation-aware recount.

This audit is deliberately read-only with respect to both inventories.  It
reports exactly how many frozen layer-8/layer-24 rows can be reused and which
new phrases would require donor extraction before the corrected tables are
adopted.

Example
-------
python stage3_jtd/compare_punctuation_inventories.py \
  --old-stage2 /data/dendritron-stage2 \
  --new-stage1 /data/dendritron-stage1-punctuation-v2/final \
  --output /data/dendritron-stage1-punctuation-v2/audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_inventory(path: Path, expected_order: int) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    expected_rank = 1
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            text = str(record["text"])
            order = int(record.get("n", expected_order))
            rank = int(record.get("rank", expected_rank))
            frequency = int(record.get("frequency", 0))
            if order != expected_order:
                raise ValueError(
                    f"{path}:{line_number}: expected n={expected_order}, found {order}"
                )
            if rank != expected_rank:
                raise ValueError(
                    f"{path}:{line_number}: expected rank {expected_rank}, found {rank}"
                )
            if text in rows:
                raise ValueError(f"{path}:{line_number}: duplicate phrase {text!r}")
            rows[text] = {"rank": rank, "frequency": frequency}
            expected_rank += 1
    if not rows:
        raise ValueError(f"Inventory is empty: {path}")
    return rows


def _write_delta(
    path: Path,
    texts: list[str],
    records: dict[str, dict[str, int]],
    *,
    order: int,
    disposition: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for text in texts:
            record = records[text]
            handle.write(
                json.dumps(
                    {
                        "text": text,
                        "frequency": int(record["frequency"]),
                        "rank": int(record["rank"]),
                        "n": int(order),
                        "disposition": disposition,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def _compare_order(
    old_path: Path,
    new_path: Path,
    output: Path,
    *,
    order: int,
    name: str,
) -> dict[str, Any]:
    old = _load_inventory(old_path, order)
    new = _load_inventory(new_path, order)
    old_keys = set(old)
    new_keys = set(new)
    reusable = old_keys & new_keys
    added = sorted(new_keys - old_keys, key=lambda text: new[text]["rank"])
    retired = sorted(old_keys - new_keys, key=lambda text: old[text]["rank"])

    _write_delta(
        output / f"{name}_new_rows.jsonl",
        added,
        new,
        order=order,
        disposition="extract_new_donor_row",
    )
    _write_delta(
        output / f"{name}_retired_rows.jsonl",
        retired,
        old,
        order=order,
        disposition="retire_old_donor_row",
    )

    denominator = len(old_keys | new_keys)
    rank_movements = [
        abs(int(old[text]["rank"]) - int(new[text]["rank"]))
        for text in reusable
    ]
    return {
        "order": order,
        "old_rows": len(old),
        "new_rows": len(new),
        "reusable_rows": len(reusable),
        "new_rows_to_extract": len(added),
        "old_rows_to_retire": len(retired),
        "overlap_fraction_of_new": len(reusable) / max(len(new), 1),
        "jaccard": len(reusable) / max(denominator, 1),
        "mean_absolute_rank_movement": (
            sum(rank_movements) / len(rank_movements) if rank_movements else 0.0
        ),
        "maximum_absolute_rank_movement": max(rank_movements, default=0),
        "sample_new_rows": [
            {"text": text, **new[text]} for text in added[:25]
        ],
        "sample_retired_rows": [
            {"text": text, **old[text]} for text in retired[:25]
        ],
        "old_file": {
            "path": str(old_path),
            "sha256": file_sha256(old_path),
        },
        "new_file": {
            "path": str(new_path),
            "sha256": file_sha256(new_path),
        },
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    reports = {
        "bigrams": _compare_order(
            args.old_stage2 / "bigrams" / "keys.jsonl",
            args.new_stage1 / "top_bigrams.jsonl",
            args.output,
            order=2,
            name="bigrams",
        ),
        "trigrams": _compare_order(
            args.old_stage2 / "trigrams" / "keys.jsonl",
            args.new_stage1 / "top_trigrams.jsonl",
            args.output,
            order=3,
            name="trigrams",
        ),
    }
    reusable = sum(item["reusable_rows"] for item in reports.values())
    total_new = sum(item["new_rows"] for item in reports.values())
    new_rows = sum(item["new_rows_to_extract"] for item in reports.values())
    retired = sum(item["old_rows_to_retire"] for item in reports.values())

    added_words: set[str] = set()
    for name in ("bigrams", "trigrams"):
        path = args.output / f"{name}_new_rows.jsonl"
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                if raw_line.strip():
                    added_words.update(json.loads(raw_line)["text"].split())

    report = {
        "schema_version": 1,
        "completed": True,
        "purpose": "punctuation_boundary_inventory_migration_audit",
        "punctuation_policy": (
            "word_ngrams_reset_at_nonwhitespace_punctuation_and_symbols; "
            "internal_apostrophes_and_hyphens_remain_inside_words"
        ),
        "orders": reports,
        "totals": {
            "corrected_rows": total_new,
            "reusable_frozen_rows": reusable,
            "new_donor_rows_to_extract": new_rows,
            "old_donor_rows_to_retire": retired,
            "reuse_fraction": reusable / max(total_new, 1),
            "distinct_words_in_new_rows": len(added_words),
        },
        "next_action": (
            "reuse matching Stage-2 rows by exact UTF-8 key and extract only "
            "the new rows, then rerun dictionary coverage and surface compilation"
        ),
    }
    atomic_write_json(args.output / "punctuation_inventory_audit.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-stage2", type=Path, required=True)
    parser.add_argument("--new-stage1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    print(json.dumps(compare(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
