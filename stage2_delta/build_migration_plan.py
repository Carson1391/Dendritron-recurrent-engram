"""Build an exact-key migration plan for the punctuation-corrected Engram bank.

The planner is deliberately tensor-free.  It compares the immutable completed
Stage-2 key tables with the corrected Stage-1 inventories and writes one plan
file per destination tensor shard.  Every corrected row is classified as:

``reuse``
    Copy the existing layer-8 and layer-24 tensors from the old row identified
    by the exact UTF-8 phrase key.

``extract``
    Run the new phrase through the locked Qwen donor and capture both layers.

Rank is a destination-table property.  A reusable phrase may move to any new
row while retaining its original paired donor vectors.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


TABLE_NAMES = {2: "bigrams", 3: "trigrams"}


def table_name(order: int) -> str:
    try:
        return TABLE_NAMES[int(order)]
    except KeyError as error:
        raise ValueError(f"Unsupported n-gram order: {order}") from error


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if raw_line.strip():
                yield line_number, json.loads(raw_line)


def load_old_key_map(
    path: Path,
    *,
    order: int,
    expected_rows: int | None = None,
) -> dict[str, dict[str, int]]:
    """Load exact UTF-8 keys and their immutable old tensor-row locations."""

    rows: dict[str, dict[str, int]] = {}
    expected_row_index = 0
    for line_number, record in iter_jsonl(path):
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number}: invalid text key")
        row_order = int(record.get("n", order))
        row_index = int(record.get("row_index", expected_row_index))
        if row_order != order:
            raise ValueError(
                f"{path}:{line_number}: expected n={order}, found {row_order}"
            )
        if row_index != expected_row_index:
            raise ValueError(
                f"{path}:{line_number}: expected row_index "
                f"{expected_row_index}, found {row_index}"
            )
        if text in rows:
            raise ValueError(f"{path}:{line_number}: duplicate phrase {text!r}")
        rows[text] = {
            "row_index": row_index,
            "donor_token_count": int(record.get("donor_token_count", 0)),
        }
        expected_row_index += 1
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows:,} rows, found {len(rows):,}"
        )
    if not rows:
        raise ValueError(f"Old key table is empty: {path}")
    return rows


def iter_corrected_inventory(
    path: Path,
    *,
    order: int,
    expected_rows: int | None = None,
) -> Iterator[dict[str, Any]]:
    expected_rank = 1
    seen_texts: set[str] = set()
    for line_number, record in iter_jsonl(path):
        text = record.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"{path}:{line_number}: invalid text key")
        row_order = int(record.get("n", order))
        rank = int(record.get("rank", expected_rank))
        frequency = int(record.get("frequency", 0))
        if row_order != order:
            raise ValueError(
                f"{path}:{line_number}: expected n={order}, found {row_order}"
            )
        if rank != expected_rank:
            raise ValueError(
                f"{path}:{line_number}: expected rank {expected_rank}, found {rank}"
            )
        if frequency < 1:
            raise ValueError(
                f"{path}:{line_number}: invalid frequency {frequency}"
            )
        if text in seen_texts:
            raise ValueError(f"{path}:{line_number}: duplicate phrase {text!r}")
        seen_texts.add(text)
        yield {
            "text": text,
            "frequency": frequency,
            "n": order,
            "source_rank": rank,
            "row_index": rank - 1,
        }
        expected_rank += 1
    found = expected_rank - 1
    if expected_rows is not None and found != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows:,} rows, found {found:,}"
        )
    if found == 0:
        raise ValueError(f"Corrected inventory is empty: {path}")


def _open_plan_temporary(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    return temporary, temporary.open("w", encoding="utf-8")


def _audit_order_counts(audit: Mapping[str, Any], name: str) -> dict[str, int]:
    try:
        report = audit["orders"][name]
        return {
            "old_rows": int(report["old_rows"]),
            "corrected_rows": int(report["new_rows"]),
            "reusable_rows": int(report["reusable_rows"]),
            "new_rows_to_extract": int(report["new_rows_to_extract"]),
            "retired_rows": int(report["old_rows_to_retire"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Audit lacks the required {name} migration counts"
        ) from error


def validate_audit_counts(
    audit: Mapping[str, Any],
    actual: Mapping[str, Mapping[str, int]],
    *,
    approved_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> None:
    """Require the audit, recomputed plan, and reviewed counts to agree."""

    for name in ("bigrams", "trigrams"):
        audited = _audit_order_counts(audit, name)
        observed = {key: int(actual[name][key]) for key in audited}
        if audited != observed:
            raise ValueError(
                f"{name} audit/plan mismatch: audit={audited}, plan={observed}"
            )
        if approved_counts is not None:
            approved = {
                key: int(value) for key, value in approved_counts[name].items()
            }
            reviewed_subset = {
                key: observed[key] for key in approved
            }
            if reviewed_subset != approved:
                raise ValueError(
                    f"{name} differs from the reviewed audit: "
                    f"approved={approved}, observed={reviewed_subset}"
                )


def build_table_plan(
    *,
    old_keys_path: Path,
    corrected_inventory_path: Path,
    plan_root: Path,
    order: int,
    old_shard_size: int,
    destination_shard_size: int,
    expected_rows: int | None = None,
) -> dict[str, Any]:
    """Write destination-shard plans in corrected frequency-rank order."""

    if old_shard_size < 1 or destination_shard_size < 1:
        raise ValueError("Shard sizes must be positive")
    name = table_name(order)
    old = load_old_key_map(
        old_keys_path,
        order=order,
        expected_rows=expected_rows,
    )

    destination = plan_root / name
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("shard-*.jsonl.tmp"):
        stale.unlink()

    reusable_rows = 0
    extracted_rows = 0
    total_rows = 0
    plan_files: list[dict[str, Any]] = []
    current_shard = -1
    current_path: Path | None = None
    current_temporary: Path | None = None
    current_handle = None
    current_rows = 0
    current_reusable = 0
    current_extract = 0

    def finish_current() -> None:
        nonlocal current_handle
        if current_handle is None or current_path is None or current_temporary is None:
            return
        current_handle.close()
        os.replace(current_temporary, current_path)
        plan_files.append(
            {
                "shard_index": current_shard,
                "path": str(current_path),
                "rows": current_rows,
                "reusable_rows": current_reusable,
                "new_rows_to_extract": current_extract,
                "sha256": file_sha256(current_path),
            }
        )
        current_handle = None

    try:
        for row in iter_corrected_inventory(
            corrected_inventory_path,
            order=order,
            expected_rows=expected_rows,
        ):
            row_index = int(row["row_index"])
            shard_index = row_index // destination_shard_size
            if shard_index != current_shard:
                finish_current()
                current_shard = shard_index
                current_rows = 0
                current_reusable = 0
                current_extract = 0
                current_path = destination / f"shard-{shard_index:05d}.jsonl"
                current_temporary, current_handle = _open_plan_temporary(current_path)

            old_row = old.get(str(row["text"]))
            if old_row is None:
                row.update(
                    {
                        "migration_disposition": "extract",
                        "source_stage2_row_index": None,
                        "source_stage2_shard_index": None,
                        "source_stage2_local_row": None,
                        "donor_token_count": 0,
                    }
                )
                extracted_rows += 1
                current_extract += 1
            else:
                old_row_index = int(old_row["row_index"])
                source_shard, source_local = divmod(old_row_index, old_shard_size)
                row.update(
                    {
                        "migration_disposition": "reuse",
                        "source_stage2_row_index": old_row_index,
                        "source_stage2_shard_index": source_shard,
                        "source_stage2_local_row": source_local,
                        "donor_token_count": int(old_row["donor_token_count"]),
                    }
                )
                reusable_rows += 1
                current_reusable += 1

            assert current_handle is not None
            current_handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            current_rows += 1
            total_rows += 1
    finally:
        finish_current()

    retired_rows = len(old) - reusable_rows
    expected_shards = math.ceil(total_rows / destination_shard_size)
    if len(plan_files) != expected_shards:
        raise RuntimeError(
            f"{name}: expected {expected_shards} plan shards, wrote {len(plan_files)}"
        )
    return {
        "order": order,
        "old_rows": len(old),
        "corrected_rows": total_rows,
        "reusable_rows": reusable_rows,
        "new_rows_to_extract": extracted_rows,
        "retired_rows": retired_rows,
        "destination_shard_size": destination_shard_size,
        "shard_count": len(plan_files),
        "plan_files": plan_files,
        "old_keys": {
            "path": str(old_keys_path),
            "sha256": file_sha256(old_keys_path),
        },
        "corrected_inventory": {
            "path": str(corrected_inventory_path),
            "sha256": file_sha256(corrected_inventory_path),
        },
    }


def build_migration_plan(
    *,
    old_stage2_root: Path,
    corrected_stage1_root: Path,
    audit_path: Path,
    output_root: Path,
    destination_shard_size: int,
    expected_rows_per_order: int | None = 500_000,
    approved_counts: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Build and fingerprint the complete two-table delta plan."""

    old_manifest_path = old_stage2_root / "manifest.json"
    if not old_manifest_path.is_file():
        raise FileNotFoundError(old_manifest_path)
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    if old_manifest.get("completed") is not True:
        raise ValueError("Original Stage-2 manifest is incomplete")
    hidden_layers = [int(value) for value in old_manifest["states"]["hidden_state_layers"]]
    if hidden_layers != [8, 24]:
        raise ValueError(f"Expected original donor layers [8, 24], found {hidden_layers}")
    if int(old_manifest["states"]["hidden_size"]) != 2048:
        raise ValueError("Expected original donor hidden size 2048")
    old_shard_size = int(old_manifest["sharding"]["shard_size"])

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("completed") is not True:
        raise ValueError("Punctuation inventory audit is incomplete")

    plan_root = output_root / "migration_plan"
    reports = {
        "bigrams": build_table_plan(
            old_keys_path=old_stage2_root / "bigrams/keys.jsonl",
            corrected_inventory_path=corrected_stage1_root / "top_bigrams.jsonl",
            plan_root=plan_root,
            order=2,
            old_shard_size=old_shard_size,
            destination_shard_size=destination_shard_size,
            expected_rows=expected_rows_per_order,
        ),
        "trigrams": build_table_plan(
            old_keys_path=old_stage2_root / "trigrams/keys.jsonl",
            corrected_inventory_path=corrected_stage1_root / "top_trigrams.jsonl",
            plan_root=plan_root,
            order=3,
            old_shard_size=old_shard_size,
            destination_shard_size=destination_shard_size,
            expected_rows=expected_rows_per_order,
        ),
    }
    validate_audit_counts(audit, reports, approved_counts=approved_counts)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "completed": True,
        "purpose": "punctuation_corrected_stage2_delta_migration",
        "identity_rule": "exact_utf8_phrase_key",
        "vector_rule": (
            "reusable rows retain paired layer08/layer24 tensors; "
            "rank only determines the corrected destination row"
        ),
        "old_stage2": {
            "root": str(old_stage2_root),
            "manifest": str(old_manifest_path),
            "manifest_sha256": file_sha256(old_manifest_path),
            "resolved_model_revision": old_manifest["model"]["resolved_revision"],
            "shard_size": old_shard_size,
        },
        "corrected_stage1_root": str(corrected_stage1_root),
        "audit": {
            "path": str(audit_path),
            "sha256": file_sha256(audit_path),
        },
        "destination": {
            "root": str(output_root),
            "shard_size": destination_shard_size,
        },
        "tables": reports,
        "totals": {
            "corrected_rows": sum(
                int(report["corrected_rows"]) for report in reports.values()
            ),
            "reusable_rows": sum(
                int(report["reusable_rows"]) for report in reports.values()
            ),
            "new_rows_to_extract": sum(
                int(report["new_rows_to_extract"]) for report in reports.values()
            ),
            "retired_rows": sum(
                int(report["retired_rows"]) for report in reports.values()
            ),
        },
    }
    manifest["fingerprint"] = json_fingerprint(manifest)
    atomic_write_json(plan_root / "manifest.json", manifest)
    return manifest


def load_plan_rows(path: Path, *, expected_order: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, row in iter_jsonl(path):
        if int(row["n"]) != expected_order:
            raise ValueError(
                f"{path}:{line_number}: expected n={expected_order}, found {row['n']}"
            )
        if row.get("migration_disposition") not in {"reuse", "extract"}:
            raise ValueError(f"{path}:{line_number}: invalid migration disposition")
        rows.append(row)
    if not rows:
        raise ValueError(f"Migration plan shard is empty: {path}")
    first = int(rows[0]["row_index"])
    for offset, row in enumerate(rows):
        if int(row["row_index"]) != first + offset:
            raise ValueError(f"{path}: discontinuous destination row sequence")
    return rows


def group_reusable_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[int, list[tuple[int, int]]]:
    """Return source-shard -> (destination-local, source-local) scatter pairs."""

    if not rows:
        return {}
    first_destination = int(rows[0]["row_index"])
    groups: dict[int, list[tuple[int, int]]] = {}
    for destination_local, row in enumerate(rows):
        if int(row["row_index"]) != first_destination + destination_local:
            raise ValueError("Destination plan rows must be contiguous")
        if row["migration_disposition"] != "reuse":
            continue
        source_shard = int(row["source_stage2_shard_index"])
        source_local = int(row["source_stage2_local_row"])
        groups.setdefault(source_shard, []).append(
            (destination_local, source_local)
        )
    return groups


def extraction_positions(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if row["migration_disposition"] == "extract"
    ]

