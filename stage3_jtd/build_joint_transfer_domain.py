"""Compile frozen Engram text keys and dictionary senses into surface keys.

This is a CPU-only stage.  It reads the immutable Stage-2 rows, loads the exact
Qwen tokenizer revision used by the donor, builds Engram's frozen canonical-ID
projection, and compiles exact phrase text into raw and complete-word lookup
artifacts. Existing donor vectors remain in their original shards and row
order.

Example
-------
python stage3_jtd/build_joint_transfer_domain.py \
  --stage2-root /data/dendritron-stage2-punctuation-v2 \
  --dictionary /data/dendritron-stage3-dictionary/inventory/dictionary.sqlite3 \
  --output /data/dendritron-stage4-jtd
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dendritron.hash_engram import HashEngramAddressor
from dendritron.jtd import (
    SURFACE_INDEX_SCHEMA_VERSION,
    compile_surface_index_database,
    file_sha256,
    iter_dictionary_records,
    iter_stage2_key_records,
)
from dendritron.tokenizer import (
    CANONICAL_PROJECTION_ALGORITHM,
    LOCKED_QWEN_TOKENIZER_ID,
    build_canonical_token_projection,
    build_tokenizer_contract,
    json_fingerprint,
    tokenizer_source_from_stage2_manifest,
)
from stage3_jtd.corrected_stage2_contract import (
    CORRECTED_JTD_ROOT,
    CORRECTED_STAGE2_ROOT,
    validate_corrected_stage2_manifest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def input_contract(
    *,
    stage2_root: Path,
    dictionary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage2_manifest_path = stage2_root / "manifest.json"
    stage2_manifest = json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
    corrected_contract = validate_corrected_stage2_manifest(
        stage2_manifest,
        manifest_path=stage2_manifest_path,
    )
    tokenizer_id, requested_revision, resolved_revision = (
        tokenizer_source_from_stage2_manifest(stage2_manifest)
    )

    bigram_keys = stage2_root / "bigrams" / "keys.jsonl"
    trigram_keys = stage2_root / "trigrams" / "keys.jsonl"
    for path in (bigram_keys, trigram_keys, dictionary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    recipient = {
        "id": tokenizer_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "add_special_tokens": False,
    }

    source = {
        "schema_version": 1,
        "surface_index_schema_version": SURFACE_INDEX_SCHEMA_VERSION,
        "canonical_projection_algorithm": CANONICAL_PROJECTION_ALGORITHM,
        "frozen_word_boundary_policy": (
            "complete_unicode_words_nonwhitespace_gap_resets_v1"
        ),
        "stage2_manifest": {
            "path": str(stage2_manifest_path),
            "sha256": file_sha256(stage2_manifest_path),
        },
        "corrected_stage2_contract": corrected_contract,
        "bigram_keys": {
            "path": str(bigram_keys),
            "sha256": file_sha256(bigram_keys),
        },
        "trigram_keys": {
            "path": str(trigram_keys),
            "sha256": file_sha256(trigram_keys),
        },
        "dictionary": {
            "path": str(dictionary_path),
            "sha256": file_sha256(dictionary_path),
        },
        "donor_tokenizer": {
            "id": tokenizer_id,
            "requested_revision": requested_revision,
            "resolved_revision": resolved_revision,
            "add_special_tokens": False,
        },
        "recipient_tokenizer": recipient,
    }
    source["fingerprint"] = json_fingerprint(source)
    return source, stage2_manifest


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    source, _stage2_manifest = input_contract(
        stage2_root=args.stage2_root,
        dictionary_path=args.dictionary,
    )
    print(f"[jtd] input contract validated — fingerprint={source['fingerprint'][:16]}", flush=True)
    final_manifest_path = args.output / "surface_index_manifest.json"
    if final_manifest_path.is_file() and not args.rebuild:
        completed = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        if (
            bool(completed.get("completed"))
            and completed.get("source", {}).get("fingerprint")
            == source["fingerprint"]
        ):
            return {
                **completed,
                "resume_action": "reused_matching_completed_jtd",
            }
        raise RuntimeError(
            "Existing JTD source fingerprint differs. Preserve the current "
            "artifact or pass --rebuild for a deliberate CPU-only rebuild."
        )

    tokenizer_info = source["recipient_tokenizer"]
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "The Qwen-recipient JTD requires transformers and tokenizers."
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(
        LOCKED_QWEN_TOKENIZER_ID,
        revision=tokenizer_info["resolved_revision"],
        cache_dir=args.cache_dir,
        trust_remote_code=False,
        use_fast=True,
    )
    print(f"[jtd] tokenizer loaded — id={tokenizer_info['id']}", flush=True)
    snapshot_root = args.output / "tokenizer_snapshot"
    tokenizer.save_pretrained(snapshot_root)
    contract = build_tokenizer_contract(
        tokenizer,
        tokenizer_id=tokenizer_info["id"],
        requested_revision=tokenizer_info["requested_revision"],
        resolved_revision=tokenizer_info["resolved_revision"],
        snapshot_root=snapshot_root,
    )
    contract_path = args.output / "tokenizer_contract.json"
    atomic_write_json(contract_path, contract.to_record())
    token_projection = build_canonical_token_projection(tokenizer)
    print(f"[jtd] canonical token projection built — {len(token_projection.raw_to_canonical)} tokens", flush=True)
    projection_path = args.output / "canonical_token_projection.json"
    atomic_write_json(
        projection_path,
        {
            **token_projection.to_record(),
            "raw_to_canonical": list(token_projection.raw_to_canonical),
        },
    )

    records = itertools.chain(
        iter_stage2_key_records(
            Path(source["bigram_keys"]["path"]),
            bank_name="bigrams",
        ),
        iter_stage2_key_records(
            Path(source["trigram_keys"]["path"]),
            bank_name="trigrams",
        ),
        iter_dictionary_records(Path(source["dictionary"]["path"])),
    )
    print("[jtd] compiling surface index database (1M+ records)...", flush=True)
    compiled = compile_surface_index_database(
        records,
        tokenizer=tokenizer,
        tokenizer_contract=contract,
        token_projection=token_projection,
        database_path=args.output / "surface_index.sqlite3",
        collision_report_path=args.output / "surface_index_collisions.jsonl",
    )
    _src_total = sum(compiled['source_rows'].values())
    print(f"[jtd] surface index compiled — {_src_total:,} source rows, {compiled['address_rows']:,} addresses", flush=True)
    manifest = {
        "schema_version": 1,
        "completed": True,
        "completed_at_utc": utc_now(),
        "source": source,
        "tokenizer_contract": {
            "path": str(contract_path),
            "sha256": file_sha256(contract_path),
            "fingerprint": contract.fingerprint,
        },
        "canonical_token_projection": {
            "path": str(projection_path),
            "sha256": file_sha256(projection_path),
            **token_projection.to_record(),
            "reference_128k_reduction_percent": 23.43,
        },
        "surface_index": compiled,
        "joint_transfer_contract": {
            "reference_frame": "frozen_qwen_layer2_definition_geometry",
            "definition_transform": "identity",
            "source_maps": ["layer8", "layer24", "live"],
            "projection_fitter": "stage3_jtd/fit_joint_transfer_domain.py",
            "surface_metadata_in_vectors": False,
        },
        "hash_engram_fallback": HashEngramAddressor().manifest_record(),
        "runtime_contract": {
            "frozen_surface_route": "longest_exact_word_order_3_2_1",
            "frozen_payload_rows": "immutable_stage2_layer08_layer24",
            "raw_language_model_ids": "exact_qwen_token_ids",
            "memory_address_ids": "canonical_projection_of_qwen_token_ids",
            "hash_engram_address_source": "canonical_qwen_token_ids_on_exact_donor_miss",
            "frozen_punctuation_policy": "offset_aligned_complete_words_with_punctuation_boundaries",
            "hash_punctuation_policy": "punctuation_preserved_as_canonical_token_classes",
            "lngram_address_source": "live_hidden_states_after_initial_context",
            "dictionary_fallback": "all_matching_sense_rows",
        },
    }
    atomic_write_json(final_manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=CORRECTED_STAGE2_ROOT,
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=Path(
            "/data/dendritron-stage3-dictionary/inventory/dictionary.sqlite3"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CORRECTED_JTD_ROOT,
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/models/huggingface"),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Deliberately rebuild the CPU tokenizer index from locked inputs",
    )
    return parser


def main() -> None:
    manifest = build(build_parser().parse_args())
    print(
        json.dumps(
            {
                "completed": manifest["completed"],
                "resume_action": manifest.get("resume_action", "built"),
                "source_fingerprint": manifest["source"]["fingerprint"],
                "tokenizer_fingerprint": manifest["tokenizer_contract"][
                    "fingerprint"
                ],
                "rows": manifest["surface_index"]["source_rows"],
                "address_rows": manifest["surface_index"]["address_rows"],
                "maximum_token_span": manifest["surface_index"]["maximum_token_span"],
                "cryptographic_hash_collisions": manifest["surface_index"][
                    "collision_report"
                ]["cryptographic_hash_collisions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
