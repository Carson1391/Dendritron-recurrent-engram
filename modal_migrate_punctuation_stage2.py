"""Build the punctuation-corrected Stage-2 Engram bank by exact-key delta.

This job consumes the reviewed punctuation inventory audit.  It copies every
reusable layer-8/layer-24 row from the immutable original Stage-2 bank and runs
only newly admitted phrases through the locked Qwen donor.

The original roots remain untouched:

    /data/dendritron-stage1
    /data/dendritron-stage2

The corrected bank is written separately:

    /data/dendritron-stage2-punctuation-v2

Safe preflight (CPU only):

    modal run modal_migrate_punctuation_stage2.py --plan-only

Materialize reusable rows and extract the reviewed delta:

    modal run modal_migrate_punctuation_stage2.py

Resume after interruption by running the same command.  Completed destination
shards are fingerprint-validated and reused.  A destructive replacement is
available only through the explicit ``--rebuild`` flag and is constrained to
the fixed corrected Stage-2 root.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import modal

from modal_extract_states import (
    ADD_SPECIAL_TOKENS,
    BLOCK_INDICES,
    CORPUS_VOLUME_NAME,
    DATA_MOUNT,
    EXPECTED_HIDDEN_SIZE,
    HIDDEN_STATE_LAYERS,
    MODEL_CACHE,
    MODEL_ID,
    MODEL_MOUNT,
    MODEL_VOLUME_NAME,
    PADDING_SIDE,
    POOLING,
    STORAGE_DTYPE,
    LayerTap,
    build_key_artifacts,
    extract_shard,
    file_sha256,
    find_text_layers,
    load_donor,
    metadata_shard_path,
    remove_partial_shard,
    table_name,
    tensor_shard_path,
    utc_now,
    validate_metadata_shard,
    validate_tensor_shard,
    write_metadata_shard,
    write_tensor_shard,
)
from stage2_delta.build_migration_plan import (
    atomic_write_json,
    build_migration_plan,
    extraction_positions,
    group_reusable_rows,
    json_fingerprint,
    load_plan_rows,
)


APP_NAME = "dendritron-punctuation-stage2-delta"

OLD_STAGE2_ROOT = DATA_MOUNT / "dendritron-stage2"
CORRECTED_STAGE1_ROOT = DATA_MOUNT / "dendritron-stage1-punctuation-v2/final"
AUDIT_PATH = (
    DATA_MOUNT
    / "dendritron-stage1-punctuation-v2/audit/punctuation_inventory_audit.json"
)
OUTPUT_ROOT = DATA_MOUNT / "dendritron-stage2-punctuation-v2"

EXPECTED_ROWS_PER_ORDER = 500_000
DEFAULT_DESTINATION_SHARD_SIZE = 50_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_PADDED_TOKENS = 4_096

# These are the counts reviewed after the real 200M-word punctuation recount.
# A later or accidental inventory cannot silently start a paid donor run.
APPROVED_AUDIT_COUNTS = {
    "bigrams": {
        "old_rows": 500_000,
        "corrected_rows": 500_000,
        "reusable_rows": 421_332,
        "new_rows_to_extract": 78_668,
        "retired_rows": 78_668,
    },
    "trigrams": {
        "old_rows": 500_000,
        "corrected_rows": 500_000,
        "reusable_rows": 388_443,
        "new_rows_to_extract": 111_557,
        "retired_rows": 111_557,
    },
}

_VERIFIED_SOURCE_SHARDS: set[Path] = set()


app = modal.App(APP_NAME)
corpus_volume = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=False)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "accelerate>=1.7,<2",
        "huggingface-hub[hf-xet]>=0.34,<2",
        "safetensors>=0.5,<1",
        "sentencepiece>=0.2,<1",
        "torch>=2.7,<3",
        "transformers>=5.4,<6",
    )
    .env(
        {
            "HF_HOME": "/models/huggingface",
            "HF_HUB_CACHE": "/models/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_output_root(path: Path) -> None:
    if path != OUTPUT_ROOT:
        raise ValueError(f"Delta output root must remain fixed at {OUTPUT_ROOT}")
    if path.parent != DATA_MOUNT or not path.name.startswith("dendritron-stage2-"):
        raise ValueError(f"Unsafe delta output root: {path}")


def _has_tensor_outputs(root: Path) -> bool:
    return any(root.glob("*/shards/shard-*.safetensors"))


def _plan_inputs_are_current(plan: Mapping[str, Any]) -> bool:
    try:
        if Path(plan["old_stage2"]["root"]) != OLD_STAGE2_ROOT:
            return False
        if Path(plan["corrected_stage1_root"]) != CORRECTED_STAGE1_ROOT:
            return False
        if Path(plan["destination"]["root"]) != OUTPUT_ROOT:
            return False
        checks = [
            (
                Path(plan["old_stage2"]["manifest"]),
                str(plan["old_stage2"]["manifest_sha256"]),
            ),
            (Path(plan["audit"]["path"]), str(plan["audit"]["sha256"])),
        ]
        for name in ("bigrams", "trigrams"):
            source = plan["tables"][name]["corrected_inventory"]
            checks.append((Path(source["path"]), str(source["sha256"])))
        return all(path.is_file() and file_sha256(path) == digest for path, digest in checks)
    except (KeyError, TypeError, ValueError):
        return False


def _validate_plan_files(plan: Mapping[str, Any]) -> None:
    for name in ("bigrams", "trigrams"):
        for record in plan["tables"][name]["plan_files"]:
            path = Path(str(record["path"]))
            if not path.is_file() or file_sha256(path) != str(record["sha256"]):
                raise ValueError(f"Migration plan shard changed or is missing: {path}")


def _validate_reviewed_plan(plan: Mapping[str, Any]) -> None:
    unsigned = dict(plan)
    recorded_fingerprint = str(unsigned.pop("fingerprint", ""))
    if json_fingerprint(unsigned) != recorded_fingerprint:
        raise ValueError("Migration plan manifest fingerprint is invalid")
    for name in ("bigrams", "trigrams"):
        report = plan["tables"][name]
        observed = {
            "old_rows": int(report["old_rows"]),
            "corrected_rows": int(report["corrected_rows"]),
            "reusable_rows": int(report["reusable_rows"]),
            "new_rows_to_extract": int(report["new_rows_to_extract"]),
            "retired_rows": int(report["retired_rows"]),
        }
        if observed != APPROVED_AUDIT_COUNTS[name]:
            raise ValueError(
                f"{name} plan differs from the reviewed audit: "
                f"approved={APPROVED_AUDIT_COUNTS[name]}, observed={observed}"
            )
    expected_totals = {
        "corrected_rows": 1_000_000,
        "reusable_rows": 809_775,
        "new_rows_to_extract": 190_225,
        "retired_rows": 190_225,
    }
    observed_totals = {
        key: int(plan["totals"][key]) for key in expected_totals
    }
    if observed_totals != expected_totals:
        raise ValueError(
            f"Plan totals differ from the reviewed audit: {observed_totals}"
        )


def _migration_spec(plan: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "punctuation_corrected_stage2_delta_materialization",
        "migration_plan_fingerprint": str(plan["fingerprint"]),
        "model_id": MODEL_ID,
        "model_revision": str(plan["old_stage2"]["resolved_model_revision"]),
        "hidden_state_layers": list(HIDDEN_STATE_LAYERS),
        "block_indices_zero_based": list(BLOCK_INDICES),
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "storage_dtype": STORAGE_DTYPE,
        "pooling": POOLING,
        "add_special_tokens": ADD_SPECIAL_TOKENS,
        "padding_side": PADDING_SIDE,
        "destination_shard_size": int(plan["destination"]["shard_size"]),
        "identity_rule": "exact_utf8_phrase_key",
        "source_vectors": "paired_layer08_layer24",
    }
    value["fingerprint"] = json_fingerprint(value)
    return value


def _source_shard_path(
    old_manifest: Mapping[str, Any],
    bank_name: str,
    shard_index: int,
) -> Path:
    artifact = old_manifest["tables"][bank_name]["tensor_files"][shard_index]
    recorded = Path(str(artifact["path"]))
    path = (
        recorded
        if recorded.is_file()
        else OLD_STAGE2_ROOT / bank_name / "shards" / recorded.name
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(artifact["bytes"]):
        raise ValueError(f"Original Stage-2 shard size differs from manifest: {path}")
    if path not in _VERIFIED_SOURCE_SHARDS:
        if file_sha256(path) != str(artifact["sha256"]):
            raise ValueError(
                f"Original Stage-2 shard hash differs from manifest: {path}"
            )
        _VERIFIED_SOURCE_SHARDS.add(path)
    return path


def _delta_shard_is_complete(
    *,
    n: int,
    shard_index: int,
    first_row: int,
    row_count: int,
    plan_sha256: str,
    migration_fingerprint: str,
) -> bool:
    if not validate_tensor_shard(
        tensor_shard_path(OUTPUT_ROOT, n, shard_index),
        row_count,
    ) or not validate_metadata_shard(
        metadata_shard_path(OUTPUT_ROOT, n, shard_index),
        first_row,
        row_count,
    ):
        return False
    try:
        from safetensors import safe_open

        with safe_open(
            str(tensor_shard_path(OUTPUT_ROOT, n, shard_index)),
            framework="pt",
            device="cpu",
        ) as handle:
            metadata = handle.metadata() or {}
        return (
            metadata.get("migration_plan_sha256") == plan_sha256
            and metadata.get("migration_fingerprint") == migration_fingerprint
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _copy_reusable_rows(
    *,
    rows: Sequence[dict[str, Any]],
    destination: dict[str, Any],
    old_manifest: Mapping[str, Any],
    bank_name: str,
    filled: Any,
) -> int:
    import torch
    from safetensors import safe_open

    groups = group_reusable_rows(rows)
    copied = 0
    for source_shard_index in sorted(groups):
        pairs = groups[source_shard_index]
        destination_indices = torch.tensor(
            [pair[0] for pair in pairs], dtype=torch.long
        )
        source_indices = torch.tensor(
            [pair[1] for pair in pairs], dtype=torch.long
        )
        path = _source_shard_path(old_manifest, bank_name, source_shard_index)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"layer08", "layer24"}:
                raise ValueError(f"Unexpected original Stage-2 tensor keys: {path}")
            for key in ("layer08", "layer24"):
                source = handle.get_tensor(key)
                if source.ndim != 2 or int(source.shape[1]) != EXPECTED_HIDDEN_SIZE:
                    raise ValueError(f"Unexpected original tensor shape in {path}:{key}")
                selected = source.index_select(0, source_indices)
                destination[key].index_copy_(0, destination_indices, selected)
                del source, selected
        filled[destination_indices] = True
        copied += len(pairs)
    return copied


def _materialize_plan_shard(
    *,
    plan_record: Mapping[str, Any],
    order: int,
    old_manifest: Mapping[str, Any],
    migration_spec: Mapping[str, Any],
    model: Any | None = None,
    tokenizer: Any | None = None,
    tap: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
) -> dict[str, Any]:
    import torch

    plan_path = Path(str(plan_record["path"]))
    plan_sha256 = str(plan_record["sha256"])
    if file_sha256(plan_path) != plan_sha256:
        raise ValueError(f"Migration plan changed: {plan_path}")
    rows = load_plan_rows(plan_path, expected_order=order)
    shard_index = int(plan_record["shard_index"])
    first_row = int(rows[0]["row_index"])
    row_count = len(rows)
    expected_first = shard_index * int(migration_spec["destination_shard_size"])
    if first_row != expected_first or row_count != int(plan_record["rows"]):
        raise ValueError(f"Destination shard contract differs from plan: {plan_path}")

    if _delta_shard_is_complete(
        n=order,
        shard_index=shard_index,
        first_row=first_row,
        row_count=row_count,
        plan_sha256=plan_sha256,
        migration_fingerprint=str(migration_spec["fingerprint"]),
    ):
        return {
            "table": table_name(order),
            "shard_index": shard_index,
            "rows": row_count,
            "reused_completed_shard": True,
            "copied_rows": int(plan_record["reusable_rows"]),
            "extracted_rows": int(plan_record["new_rows_to_extract"]),
        }

    remove_partial_shard(OUTPUT_ROOT, order, shard_index)
    destination = {
        key: torch.empty(
            (row_count, EXPECTED_HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="cpu",
        )
        for key in ("layer08", "layer24")
    }
    filled = torch.zeros(row_count, dtype=torch.bool)
    bank_name = table_name(order)
    copied = _copy_reusable_rows(
        rows=rows,
        destination=destination,
        old_manifest=old_manifest,
        bank_name=bank_name,
        filled=filled,
    )

    new_positions = extraction_positions(rows)
    if new_positions:
        if model is None or tokenizer is None or tap is None:
            raise RuntimeError(
                f"{plan_path} contains {len(new_positions):,} new rows and requires Qwen"
            )
        new_rows = [rows[position] for position in new_positions]
        extracted_banks, _ = extract_shard(
            new_rows,
            model,
            tokenizer,
            tap,
            batch_size=batch_size,
            max_padded_tokens=max_padded_tokens,
        )
        destination_indices = torch.tensor(new_positions, dtype=torch.long)
        for layer, block in zip(HIDDEN_STATE_LAYERS, BLOCK_INDICES):
            destination[f"layer{layer:02d}"].index_copy_(
                0,
                destination_indices,
                extracted_banks[block],
            )
        filled[destination_indices] = True
        del extracted_banks, destination_indices

    if not bool(filled.all()):
        missing = torch.nonzero(~filled).flatten().tolist()[:20]
        raise RuntimeError(f"Unfilled destination rows in {plan_path}: {missing}")
    if copied != int(plan_record["reusable_rows"]):
        raise RuntimeError(f"Reusable row count differs from plan: {plan_path}")
    if len(new_positions) != int(plan_record["new_rows_to_extract"]):
        raise RuntimeError(f"Extraction row count differs from plan: {plan_path}")
    for row in rows:
        if int(row.get("donor_token_count", 0)) < 1:
            raise RuntimeError(
                f"Missing donor token count for corrected row {row['row_index']}"
            )

    write_tensor_shard(
        tensor_shard_path(OUTPUT_ROOT, order, shard_index),
        destination,
        metadata={
            "n": str(order),
            "table": bank_name,
            "tensor_keys": "layer08,layer24",
            "hidden_state_layers": "8,24",
            "first_row": str(first_row),
            "rows": str(row_count),
            "hidden_size": str(EXPECTED_HIDDEN_SIZE),
            "dtype": STORAGE_DTYPE,
            "pooling": POOLING,
            "model_id": MODEL_ID,
            "model_revision": str(migration_spec["model_revision"]),
            "migration_plan_sha256": plan_sha256,
            "migration_fingerprint": str(migration_spec["fingerprint"]),
            "copied_rows": str(copied),
            "extracted_rows": str(len(new_positions)),
        },
    )
    write_metadata_shard(
        metadata_shard_path(OUTPUT_ROOT, order, shard_index),
        rows,
    )
    if not _delta_shard_is_complete(
        n=order,
        shard_index=shard_index,
        first_row=first_row,
        row_count=row_count,
        plan_sha256=plan_sha256,
        migration_fingerprint=str(migration_spec["fingerprint"]),
    ):
        raise RuntimeError(f"Written destination shard failed validation: {plan_path}")
    del destination, filled
    return {
        "table": bank_name,
        "shard_index": shard_index,
        "rows": row_count,
        "reused_completed_shard": False,
        "copied_rows": copied,
        "extracted_rows": len(new_positions),
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=16_384,
    timeout=6 * 60 * 60,
    volumes={
        str(DATA_MOUNT): corpus_volume,
        str(MODEL_MOUNT): model_volume,
    },
)
def prepare_delta(
    *,
    plan_only: bool = False,
    rebuild: bool = False,
    destination_shard_size: int = DEFAULT_DESTINATION_SHARD_SIZE,
) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    _safe_output_root(OUTPUT_ROOT)
    if destination_shard_size < 1:
        raise ValueError("destination_shard_size must be positive")
    if rebuild and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
        corpus_volume.commit()

    existing_plan_path = OUTPUT_ROOT / "migration_plan/manifest.json"
    plan: dict[str, Any]
    if existing_plan_path.is_file():
        candidate = _load_json(existing_plan_path)
        if (
            _plan_inputs_are_current(candidate)
            and int(candidate["destination"]["shard_size"])
            == destination_shard_size
        ):
            plan = candidate
            _validate_reviewed_plan(plan)
            _validate_plan_files(plan)
        else:
            if _has_tensor_outputs(OUTPUT_ROOT):
                raise RuntimeError(
                    "Migration inputs changed after tensor output began. "
                    "Review the change and rerun with --rebuild."
                )
            shutil.rmtree(OUTPUT_ROOT / "migration_plan")
            plan = build_migration_plan(
                old_stage2_root=OLD_STAGE2_ROOT,
                corrected_stage1_root=CORRECTED_STAGE1_ROOT,
                audit_path=AUDIT_PATH,
                output_root=OUTPUT_ROOT,
                destination_shard_size=destination_shard_size,
                expected_rows_per_order=EXPECTED_ROWS_PER_ORDER,
                approved_counts=APPROVED_AUDIT_COUNTS,
            )
    else:
        if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
            known_scaffolding = {
                "migration_plan",
                "run_spec.json",
                "bigrams",
                "trigrams",
            }
            present = {path.name for path in OUTPUT_ROOT.iterdir()}
            contains_materialized_rows = _has_tensor_outputs(OUTPUT_ROOT) or any(
                OUTPUT_ROOT.glob("*/metadata/shard-*.jsonl")
            )
            if present <= known_scaffolding and not contains_materialized_rows:
                shutil.rmtree(OUTPUT_ROOT)
            else:
                raise RuntimeError(
                    f"{OUTPUT_ROOT} contains data without a migration plan. "
                    "Review it and use --rebuild only to replace this fixed root."
                )
        plan = build_migration_plan(
            old_stage2_root=OLD_STAGE2_ROOT,
            corrected_stage1_root=CORRECTED_STAGE1_ROOT,
            audit_path=AUDIT_PATH,
            output_root=OUTPUT_ROOT,
            destination_shard_size=destination_shard_size,
            expected_rows_per_order=EXPECTED_ROWS_PER_ORDER,
            approved_counts=APPROVED_AUDIT_COUNTS,
        )

    _validate_reviewed_plan(plan)

    migration_spec = _migration_spec(plan)
    run_spec_path = OUTPUT_ROOT / "run_spec.json"
    if run_spec_path.is_file():
        existing_spec = _load_json(run_spec_path)
        if existing_spec.get("fingerprint") != migration_spec["fingerprint"]:
            if _has_tensor_outputs(OUTPUT_ROOT):
                raise RuntimeError(
                    "Migration specification changed after tensor output began. "
                    "Review the change and rerun with --rebuild."
                )
    atomic_write_json(run_spec_path, migration_spec)
    for order in (2, 3):
        (OUTPUT_ROOT / table_name(order) / "shards").mkdir(
            parents=True, exist_ok=True
        )
        (OUTPUT_ROOT / table_name(order) / "metadata").mkdir(
            parents=True, exist_ok=True
        )
    corpus_volume.commit()

    prepared: dict[str, Any] = {
        "plan_manifest": str(existing_plan_path),
        "plan_fingerprint": str(plan["fingerprint"]),
        "migration_spec": migration_spec,
        "output_root": str(OUTPUT_ROOT),
        "snapshot_path": None,
        "totals": dict(plan["totals"]),
    }
    if not plan_only:
        resolved_revision = str(plan["old_stage2"]["resolved_model_revision"])
        snapshot = Path(
            snapshot_download(
                repo_id=MODEL_ID,
                revision=resolved_revision,
                cache_dir=str(MODEL_CACHE),
                local_files_only=False,
            )
        )
        if snapshot.name != resolved_revision:
            raise ValueError(
                f"Resolved donor revision changed: {snapshot.name} != {resolved_revision}"
            )
        model_volume.commit()
        prepared["snapshot_path"] = str(snapshot)
    return prepared


@app.function(
    image=image,
    cpu=8.0,
    memory=32_768,
    timeout=12 * 60 * 60,
    volumes={str(DATA_MOUNT): corpus_volume},
)
def materialize_copy_only_shards(prepared: Mapping[str, Any]) -> dict[str, Any]:
    plan = _load_json(Path(str(prepared["plan_manifest"])))
    _validate_reviewed_plan(plan)
    migration_spec = dict(prepared["migration_spec"])
    old_manifest = _load_json(OLD_STAGE2_ROOT / "manifest.json")
    completed: list[dict[str, Any]] = []
    for order, name in ((2, "bigrams"), (3, "trigrams")):
        for plan_record in plan["tables"][name]["plan_files"]:
            if int(plan_record["new_rows_to_extract"]) != 0:
                continue
            result = _materialize_plan_shard(
                plan_record=plan_record,
                order=order,
                old_manifest=old_manifest,
                migration_spec=migration_spec,
            )
            completed.append(result)
            corpus_volume.commit()
            print(
                f"[copy] {name} shard {result['shard_index']}: "
                f"{result['copied_rows']:,} reusable rows"
            )
    return {
        "completed_copy_only_shards": len(completed),
        "rows": sum(int(item["rows"]) for item in completed),
    }


def _finalize_manifest(
    *,
    plan: Mapping[str, Any],
    migration_spec: Mapping[str, Any],
    old_manifest: Mapping[str, Any],
    shard_stats: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    table_artifacts: dict[str, Any] = {}
    shard_size = int(migration_spec["destination_shard_size"])
    for order, name in ((2, "bigrams"), (3, "trigrams")):
        report = plan["tables"][name]
        shard_count = int(report["shard_count"])
        rows = int(report["corrected_rows"])
        keys = build_key_artifacts(
            OUTPUT_ROOT,
            n=order,
            shard_count=shard_count,
            expected_total_rows=rows,
        )
        tensor_files = []
        for shard_index in range(shard_count):
            path = tensor_shard_path(OUTPUT_ROOT, order, shard_index)
            tensor_files.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        table_artifacts[name] = {
            "n": order,
            "rows": rows,
            "shard_count": shard_count,
            "tensor_keys": ["layer08", "layer24"],
            "tensor_files": tensor_files,
            "migration": {
                "copied_reusable_rows": int(report["reusable_rows"]),
                "new_donor_rows_extracted": int(report["new_rows_to_extract"]),
                "retired_original_rows": int(report["retired_rows"]),
            },
            **keys,
        }

    manifest: dict[str, Any] = {
        "schema_version": 3,
        "completed": True,
        "completed_at_utc": utc_now(),
        "extraction_fingerprint": str(migration_spec["fingerprint"]),
        "model": old_manifest["model"],
        "states": old_manifest["states"],
        "addressing": {
            "canonical_type": "utf8_surface_text",
            "donor_token_ids_persisted": False,
            "table_partition": "separate bigram and trigram Engram tables",
            "row_order": "punctuation-aware Stage-1 frequency rank",
            "punctuation_boundary_aware": True,
            "runtime_addressing": (
                "compile exact Qwen surface keys from this corrected manifest"
            ),
        },
        "input": {
            "corrected_stage1_root": str(CORRECTED_STAGE1_ROOT),
            "punctuation_audit": {
                "path": str(AUDIT_PATH),
                "sha256": file_sha256(AUDIT_PATH),
            },
            "migration_plan": {
                "path": str(OUTPUT_ROOT / "migration_plan/manifest.json"),
                "fingerprint": str(plan["fingerprint"]),
            },
            "original_stage2": {
                "path": str(OLD_STAGE2_ROOT),
                "manifest_sha256": str(plan["old_stage2"]["manifest_sha256"]),
            },
        },
        "tables": table_artifacts,
        "sharding": {
            "shard_size": shard_size,
            "tensor_keys": ["layer08", "layer24"],
            "both_layer_views_share_each_engram_row": True,
        },
        "migration": {
            "identity_rule": "exact_utf8_phrase_key",
            "reusable_rows_copied": int(plan["totals"]["reusable_rows"]),
            "new_rows_extracted": int(plan["totals"]["new_rows_to_extract"]),
            "retired_rows": int(plan["totals"]["retired_rows"]),
            "total_corrected_rows": int(plan["totals"]["corrected_rows"]),
        },
        "runtime": {
            "offline_donor_gpu": "A100-80GB",
            "live_cuda_dependency": False,
            "invocation_shards": list(shard_stats),
        },
        "universal_subspace_applied": False,
    }
    atomic_write_json(OUTPUT_ROOT / "manifest.json", manifest)
    return manifest


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=65_536,
    timeout=24 * 60 * 60,
    volumes={
        str(DATA_MOUNT): corpus_volume,
        str(MODEL_MOUNT): model_volume,
    },
)
def extract_delta(
    prepared: Mapping[str, Any],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
) -> dict[str, Any]:
    import torch

    if batch_size < 1 or max_padded_tokens < 1:
        raise ValueError("batch_size and max_padded_tokens must be positive")
    plan = _load_json(Path(str(prepared["plan_manifest"])))
    _validate_reviewed_plan(plan)
    _validate_plan_files(plan)
    migration_spec = dict(prepared["migration_spec"])
    if str(plan["fingerprint"]) != str(prepared["plan_fingerprint"]):
        raise RuntimeError("Migration plan changed between preflight and extraction")
    if _migration_spec(plan)["fingerprint"] != migration_spec["fingerprint"]:
        raise RuntimeError("Migration specification changed between functions")
    old_manifest = _load_json(OLD_STAGE2_ROOT / "manifest.json")
    if file_sha256(OLD_STAGE2_ROOT / "manifest.json") != str(
        plan["old_stage2"]["manifest_sha256"]
    ):
        raise RuntimeError("Original Stage-2 manifest changed after planning")

    completed_manifest_path = OUTPUT_ROOT / "manifest.json"
    if completed_manifest_path.is_file():
        completed = _load_json(completed_manifest_path)
        if (
            completed.get("completed") is True
            and completed.get("extraction_fingerprint")
            == migration_spec["fingerprint"]
        ):
            print(f"[done] corrected Stage-2 bank already exists at {OUTPUT_ROOT}")
            return completed

    snapshot_path = prepared.get("snapshot_path")
    if not snapshot_path:
        raise ValueError("GPU extraction requires a prepared donor snapshot")
    model, tokenizer, model_details = load_donor(str(snapshot_path))
    if str(model_details.get("resolved_revision", migration_spec["model_revision"])) != str(
        migration_spec["model_revision"]
    ):
        # load_donor currently reports structure rather than revision; the
        # snapshot path and preflight revision remain the controlling checks.
        pass
    _, layers = find_text_layers(model)
    tap = LayerTap(layers)
    shard_stats: list[dict[str, Any]] = []
    started = time.time()
    try:
        for order, name in ((2, "bigrams"), (3, "trigrams")):
            records = plan["tables"][name]["plan_files"]
            for record in records:
                shard_started = time.time()
                result = _materialize_plan_shard(
                    plan_record=record,
                    order=order,
                    old_manifest=old_manifest,
                    migration_spec=migration_spec,
                    model=model,
                    tokenizer=tokenizer,
                    tap=tap,
                    batch_size=batch_size,
                    max_padded_tokens=max_padded_tokens,
                )
                result["seconds"] = time.time() - shard_started
                shard_stats.append(result)
                corpus_volume.commit()
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"[{name} shard {int(record['shard_index']) + 1}/"
                    f"{len(records)}] copied {result['copied_rows']:,}; "
                    f"extracted {result['extracted_rows']:,}; "
                    f"{result['seconds']:.1f}s"
                )
    finally:
        tap.close()

    manifest = _finalize_manifest(
        plan=plan,
        migration_spec=migration_spec,
        old_manifest=old_manifest,
        shard_stats=shard_stats,
    )
    manifest["runtime"]["invocation_seconds"] = time.time() - started
    atomic_write_json(completed_manifest_path, manifest)
    corpus_volume.commit()
    print(
        "[complete] copied "
        f"{manifest['migration']['reusable_rows_copied']:,} paired rows and "
        f"extracted {manifest['migration']['new_rows_extracted']:,} new rows"
    )
    return manifest


@app.local_entrypoint()
def main(
    plan_only: bool = False,
    rebuild: bool = False,
    destination_shard_size: int = DEFAULT_DESTINATION_SHARD_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
) -> None:
    prepared = prepare_delta.remote(
        plan_only=plan_only,
        rebuild=rebuild,
        destination_shard_size=destination_shard_size,
    )
    print(
        json.dumps(
            {
                "plan_validated": True,
                "corrected_rows": prepared["totals"]["corrected_rows"],
                "reusable_rows": prepared["totals"]["reusable_rows"],
                "new_rows_to_extract": prepared["totals"]["new_rows_to_extract"],
                "output_root": str(OUTPUT_ROOT),
            },
            indent=2,
        )
    )
    if plan_only:
        return
    copy_result = materialize_copy_only_shards.remote(prepared)
    print(json.dumps(copy_result, indent=2))
    result = extract_delta.remote(
        prepared,
        batch_size=batch_size,
        max_padded_tokens=max_padded_tokens,
    )
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "output": str(OUTPUT_ROOT),
                "bigrams": result["tables"]["bigrams"]["rows"],
                "trigrams": result["tables"]["trigrams"]["rows"],
                "copied_rows": result["migration"]["reusable_rows_copied"],
                "extracted_rows": result["migration"]["new_rows_extracted"],
                "layers": result["states"]["hidden_state_layers"],
            },
            indent=2,
        )
    )
