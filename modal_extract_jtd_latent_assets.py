"""Export Dendritron's learned Qwen symbols and real JTD anchor pairs.

This is the bounded offline donor step after the dictionary layer-2 bank:

1. Export Qwen's full input-embedding table.  Every raw tokenizer symbol is
   covered, including punctuation, whitespace, subword pieces, and specials.
2. Select deterministic phrase rows from both completed Engram banks.
3. Append the dictionary's fixed ``\nConcept:`` readout marker to the exact
   phrase text and run it through Qwen block 1 / ``hidden_states[2]``.
4. Store row-aligned ``layer08``, ``layer24``, and ``layer02`` anchor triples.

The surface words and IDs remain metadata.  The numerical triples contain only
hidden states.  The resulting source maps are fitted on CPU by
``stage3_jtd/fit_joint_transfer_domain.py``.

Run a small contract check first:

    modal run modal_extract_jtd_latent_assets.py --smoke

Then build the real assets:

    modal run modal_extract_jtd_latent_assets.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import unicodedata
from typing import Any, Iterable, Sequence

import modal

from modal_extract_states import (
    CORPUS_VOLUME_NAME,
    DATA_MOUNT,
    EXPECTED_HIDDEN_SIZE,
    MODEL_CACHE,
    MODEL_ID,
    MODEL_MOUNT,
    MODEL_VOLUME_NAME,
)
from dendritron.definition_bank import DEFINITION_READOUT_MARKER


APP_NAME = "dendritron-jtd-latent-assets"
STAGE2_ROOT = "/data/dendritron-stage2-punctuation-v2"
DICTIONARY_MANIFEST = "/data/dendritron-stage3-dictionary/bank/manifest.json"
OUTPUT_ROOT = "/data/dendritron-stage4-jtd-punctuation-v2/latent-assets"
SMOKE_OUTPUT_ROOT = "/data/dendritron-stage4-jtd-punctuation-v2-smoke/latent-assets"

DEFAULT_ANCHORS_PER_ORDER = 32_768
SMOKE_ANCHORS_PER_ORDER = 128
DEFAULT_BATCH_SIZE = 512
DEFAULT_MAX_PADDED_TOKENS = 16_384

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
    .add_local_file(
        local_path=Path(__file__).parent / "modal_extract_states.py",
        remote_path="/root/modal_extract_states.py",
    )
    .add_local_file(
        local_path=Path(__file__).parent / "modal_extract_definition_states.py",
        remote_path="/root/modal_extract_definition_states.py",
    )
    .add_local_dir(
        local_path=Path(__file__).parent / "stage3_jtd",
        remote_path="/root/stage3_jtd",
    )
    .add_local_dir(
        local_path=Path(__file__).parent / "dendritron",
        remote_path="/root/dendritron",
    )
)


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


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def deterministic_anchor_indices(total_rows: int, requested: int) -> tuple[int, ...]:
    """Keep a frequent head plus an evenly spaced sample of the full rank tail."""

    if total_rows < 1:
        raise ValueError("total_rows must be positive")
    if requested < 1 or requested >= total_rows:
        return tuple(range(total_rows))
    head = max(1, min(requested // 4, 8_192))
    selected = set(range(head))
    tail_needed = requested - len(selected)
    if tail_needed == 1:
        selected.add(total_rows - 1)
    elif tail_needed > 1:
        span = total_rows - head - 1
        for position in range(tail_needed):
            selected.add(head + round(position * span / (tail_needed - 1)))
    candidate = head
    while len(selected) < requested:
        selected.add(candidate)
        candidate += 1
    return tuple(sorted(selected)[:requested])


def _selected_key_rows(
    path: Path,
    selected_indices: Sequence[int],
    *,
    expected_order: int,
) -> list[dict[str, Any]]:
    wanted = set(int(value) for value in selected_indices)
    rows: list[dict[str, Any]] = []
    next_row = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            row_index = int(record.get("row_index", next_row))
            if row_index != next_row:
                raise ValueError(
                    f"{path}: expected row {next_row}, found {row_index}"
                )
            if int(record.get("n", expected_order)) != expected_order:
                raise ValueError(f"{path}: incorrect n-gram order at row {row_index}")
            if row_index in wanted:
                surface_text = str(record["text"])
                rows.append(
                    {
                        "row_index": row_index,
                        "text": surface_text,
                        "donor_text": surface_text + DEFINITION_READOUT_MARKER,
                        "frequency": int(record.get("frequency", 0)),
                    }
                )
            next_row += 1
    if len(rows) != len(selected_indices):
        raise ValueError(
            f"{path}: selected {len(rows)} of {len(selected_indices)} requested rows"
        )
    return rows


def _batches(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    batch_size: int,
    max_padded_tokens: int,
) -> Iterable[list[dict[str, Any]]]:
    from modal_extract_definition_states import tokenizer_lengths

    pending: list[dict[str, Any]] = []
    lengths: list[int] = []
    for row in rows:
        length = tokenizer_lengths(tokenizer, [str(row["donor_text"])])[0]
        proposed_max = max(lengths + [length])
        proposed_count = len(pending) + 1
        if pending and (
            proposed_count > batch_size
            or proposed_count * proposed_max > max_padded_tokens
        ):
            yield pending
            pending = []
            lengths = []
        pending.append(row)
        lengths.append(length)
    if pending:
        yield pending


def _stage2_shard_path(
    stage2_root: Path,
    stage2_manifest: dict[str, Any],
    bank_name: str,
    shard_index: int,
) -> Path:
    artifact = stage2_manifest["tables"][bank_name]["tensor_files"][shard_index]
    recorded = Path(str(artifact["path"]))
    path = (
        recorded
        if recorded.is_file()
        else stage2_root / bank_name / "shards" / recorded.name
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(artifact["bytes"]):
        raise ValueError(f"Stage-2 shard size differs from manifest: {path}")
    return path


def _load_selected_engram_rows(
    stage2_root: Path,
    stage2_manifest: dict[str, Any],
    bank_name: str,
    indices: Sequence[int],
) -> tuple[Any, Any]:
    import torch
    from safetensors import safe_open

    shard_size = int(stage2_manifest["sharding"]["shard_size"])
    grouped: dict[int, list[int]] = {}
    for row_index in indices:
        shard_index, local_row = divmod(int(row_index), shard_size)
        grouped.setdefault(shard_index, []).append(local_row)

    layer8_batches = []
    layer24_batches = []
    for shard_index in sorted(grouped):
        path = _stage2_shard_path(
            stage2_root,
            stage2_manifest,
            bank_name,
            shard_index,
        )
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"layer08", "layer24"}:
                raise ValueError(f"Unexpected Stage-2 tensor keys: {path}")
            local = torch.tensor(grouped[shard_index], dtype=torch.long)
            layer8_batches.append(handle.get_tensor("layer08").index_select(0, local))
            layer24_batches.append(handle.get_tensor("layer24").index_select(0, local))
    return torch.cat(layer8_batches), torch.cat(layer24_batches)


def _token_symbol_statistics(tokenizer: Any, rows: int) -> dict[str, int]:
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    counts = {
        "rows": rows,
        "special": 0,
        "whitespace_only": 0,
        "punctuation_or_symbol_only": 0,
        "contains_letter_or_number": 0,
        "other": 0,
    }
    for token_id in range(rows):
        if token_id in special_ids:
            counts["special"] += 1
        decoded = str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        if decoded and decoded.isspace():
            counts["whitespace_only"] += 1
            continue
        visible = [character for character in decoded if not character.isspace()]
        if any(unicodedata.category(character)[:1] in {"L", "N"} for character in visible):
            counts["contains_letter_or_number"] += 1
        elif visible and all(
            unicodedata.category(character)[:1] in {"P", "S"}
            for character in visible
        ):
            counts["punctuation_or_symbol_only"] += 1
        else:
            counts["other"] += 1
    return counts


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
def prepare_assets(stage2_root: str = str(STAGE2_ROOT)) -> dict[str, Any]:
    from huggingface_hub import snapshot_download
    from stage3_jtd.corrected_stage2_contract import (
        validate_corrected_stage2_manifest,
    )

    resolved_stage2_root = Path(stage2_root)
    stage2_manifest_path = resolved_stage2_root / "manifest.json"
    stage2 = json.loads(stage2_manifest_path.read_text(encoding="utf-8"))
    corrected_contract = validate_corrected_stage2_manifest(
        stage2,
        manifest_path=stage2_manifest_path,
    )
    resolved_revision = str(stage2["model"]["resolved_revision"])
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=resolved_revision,
            cache_dir=str(MODEL_CACHE),
            local_files_only=False,
        )
    )
    model_volume.commit()
    return {
        "snapshot_path": str(snapshot),
        "resolved_revision": snapshot.name,
        "stage2_root": str(resolved_stage2_root),
        "stage2_manifest": str(stage2_manifest_path),
        "stage2_manifest_sha256": file_sha256(stage2_manifest_path),
        "corrected_stage2_contract": corrected_contract,
    }


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
def extract_latent_assets(
    assets: dict[str, Any],
    smoke: bool = False,
    rebuild: bool = False,
    anchors_per_order: int = DEFAULT_ANCHORS_PER_ORDER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
    output_root: str = str(OUTPUT_ROOT),
) -> dict[str, Any]:
    import torch
    from safetensors.torch import save_file
    from modal_extract_definition_states import (
        Layer02Tap,
        encode_layer02,
        find_text_layers,
        load_definition_donor,
    )

    stage2_path = Path(assets["stage2_manifest"])
    stage2_root = Path(assets["stage2_root"])
    stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
    dictionary = json.loads(Path(DICTIONARY_MANIFEST).read_text(encoding="utf-8"))
    if int(dictionary.get("hidden_state_layer", -1)) != 2:
        raise ValueError("Dictionary bank must use Qwen hidden_states[2]")
    if int(dictionary.get("hidden_size", -1)) != EXPECTED_HIDDEN_SIZE:
        raise ValueError("Dictionary hidden width differs from Stage 2")
    if str(dictionary.get("resolved_revision")) != str(assets["resolved_revision"]):
        raise ValueError("Dictionary and Stage-2 donor revisions differ")

    root = Path(SMOKE_OUTPUT_ROOT) if smoke else Path(output_root)
    if not root.is_absolute() or root.parent.parent != DATA_MOUNT:
        raise ValueError(
            "Latent-asset output must be /data/<stage-directory>/<child>"
        )
    requested = SMOKE_ANCHORS_PER_ORDER if smoke else int(anchors_per_order)
    source_contract = {
        "schema_version": 1,
        "stage2_manifest_sha256": file_sha256(stage2_path),
        "corrected_stage2_contract": assets["corrected_stage2_contract"],
        "dictionary_manifest_sha256": file_sha256(Path(DICTIONARY_MANIFEST)),
        "resolved_revision": str(assets["resolved_revision"]),
        "reference_hidden_state_layer": 2,
        "reference_readout_protocol": "same_phrase_plus_dictionary_marker_v1",
        "reference_readout_marker": DEFINITION_READOUT_MARKER,
        "reference_pooling": "final_non_padding_marker_token",
        "source_hidden_state_layers": [8, 24],
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "anchors_per_order": requested,
        "anchor_selection": "frequent_quarter_plus_even_full_rank_tail_v1",
        "surface_metadata_in_vectors": False,
        "token_embedding_source": "qwen_input_embedding_weight",
        "smoke": bool(smoke),
    }
    source_contract["fingerprint"] = json_fingerprint(source_contract)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not rebuild:
        completed = json.loads(manifest_path.read_text(encoding="utf-8"))
        if completed.get("source_contract", {}).get("fingerprint") == source_contract["fingerprint"]:
            return {**completed, "resume_action": "reused_matching_completed_assets"}
        raise RuntimeError("Existing latent assets use a different source contract")
    if rebuild and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_details = load_definition_donor(assets["snapshot_path"])
    marker_token_ids = tokenizer(
        DEFINITION_READOUT_MARKER,
        add_special_tokens=False,
    )["input_ids"]
    if not marker_token_ids:
        raise ValueError("The fixed layer-2 readout marker tokenized to zero IDs")
    _layer_path, layers = find_text_layers(model)
    tap = Layer02Tap(layers[1])
    try:
        embedding_module = model.get_input_embeddings()
        if embedding_module is None or not hasattr(embedding_module, "weight"):
            raise RuntimeError("Qwen model does not expose input embeddings")
        token_embeddings = embedding_module.weight.detach().to(
            device="cpu",
            dtype=torch.bfloat16,
        )
        if token_embeddings.ndim != 2 or token_embeddings.shape[1] != EXPECTED_HIDDEN_SIZE:
            raise ValueError("Unexpected Qwen input-embedding shape")
        if token_embeddings.shape[0] < len(tokenizer):
            raise ValueError(
                "Embedding matrix smaller than tokenizer vocabulary: "
                f"{token_embeddings.shape[0]} versus {len(tokenizer)}"
            )
        if token_embeddings.shape[0] > len(tokenizer):
            token_embeddings = token_embeddings[:len(tokenizer)].contiguous()
        frontend_root = root / "frontend"
        frontend_root.mkdir(parents=True, exist_ok=True)
        token_path = frontend_root / "qwen_token_embeddings.safetensors"
        token_tmp = token_path.with_name(token_path.name + ".tmp")
        save_file(
            {"token_embeddings": token_embeddings.contiguous()},
            str(token_tmp),
            metadata={
                "model_id": MODEL_ID,
                "resolved_revision": str(assets["resolved_revision"]),
                "includes_punctuation": "true",
            },
        )
        os.replace(token_tmp, token_path)
        symbol_stats = _token_symbol_statistics(tokenizer, token_embeddings.shape[0])
        del token_embeddings

        anchor_artifacts: dict[str, Any] = {}
        for bank_name, order in (("bigrams", 2), ("trigrams", 3)):
            total_rows = int(stage2["tables"][bank_name]["rows"])
            indices = deterministic_anchor_indices(total_rows, requested)
            rows = _selected_key_rows(
                stage2_root / bank_name / "keys.jsonl",
                indices,
                expected_order=order,
            )
            references = []
            for batch in _batches(
                rows,
                tokenizer,
                batch_size=batch_size,
                max_padded_tokens=max_padded_tokens,
            ):
                references.append(encode_layer02(batch, model, tokenizer, tap))
            layer02 = torch.cat(references).contiguous()
            layer08, layer24 = _load_selected_engram_rows(
                stage2_root,
                stage2,
                bank_name,
                indices,
            )
            if not (
                layer02.shape == layer08.shape == layer24.shape
                == (len(indices), EXPECTED_HIDDEN_SIZE)
            ):
                raise ValueError(f"JTD anchor shape mismatch for {bank_name}")
            anchor_path = root / "anchors" / f"{bank_name}.safetensors"
            anchor_path.parent.mkdir(parents=True, exist_ok=True)
            anchor_tmp = anchor_path.with_name(anchor_path.name + ".tmp")
            save_file(
                {
                    "row_indices": torch.tensor(indices, dtype=torch.int64),
                    "layer02": layer02,
                    "layer08": layer08,
                    "layer24": layer24,
                },
                str(anchor_tmp),
                metadata={
                    "bank_name": bank_name,
                    "word_order": str(order),
                    "layer02_readout_protocol": (
                        "same_phrase_plus_dictionary_marker_v1"
                    ),
                    "layer02_readout_marker": DEFINITION_READOUT_MARKER,
                    "layer02_pooling": "final_non_padding_marker_token",
                    "surface_metadata_in_vectors": "false",
                },
            )
            os.replace(anchor_tmp, anchor_path)
            anchor_artifacts[bank_name] = {
                "path": str(anchor_path),
                "sha256": file_sha256(anchor_path),
                "bytes": anchor_path.stat().st_size,
                "rows": len(indices),
                "first_source_row": indices[0],
                "last_source_row": indices[-1],
            }
            del rows, references, layer02, layer08, layer24
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        tap.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "completed": True,
        "source_contract": source_contract,
        "model_details": model_details,
        "frontend": {
            "token_embeddings": {
                "path": str(token_path),
                "sha256": file_sha256(token_path),
                "bytes": token_path.stat().st_size,
                "shape": [symbol_stats["rows"], EXPECTED_HIDDEN_SIZE],
                "dtype": "bfloat16",
            },
            "symbol_statistics": symbol_stats,
            "punctuation_contract": (
                "every raw Qwen punctuation/symbol row is present in the learned "
                "input table and remains an output-vocabulary symbol"
            ),
        },
        "anchors": anchor_artifacts,
        "anchor_reference": {
            "hidden_state_layer": 2,
            "readout_protocol": "same_phrase_plus_dictionary_marker_v1",
            "readout_marker": DEFINITION_READOUT_MARKER,
            "readout_marker_token_ids": [int(value) for value in marker_token_ids],
            "pooling": "final_non_padding_marker_token",
        },
        "definition_reference": {
            "transform": "identity",
            "manifest": str(DICTIONARY_MANIFEST),
            "manifest_sha256": file_sha256(Path(DICTIONARY_MANIFEST)),
        },
        "live_map_status": "identity_initialization_then_recipient_training",
    }
    atomic_write_json(manifest_path, manifest)
    corpus_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    smoke: bool = False,
    rebuild: bool = False,
    anchors_per_order: int = DEFAULT_ANCHORS_PER_ORDER,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
    stage2_root: str = str(STAGE2_ROOT),
    output_root: str = str(OUTPUT_ROOT),
) -> None:
    assets = prepare_assets.remote(stage2_root=stage2_root)
    result = extract_latent_assets.remote(
        assets,
        smoke=smoke,
        rebuild=rebuild,
        anchors_per_order=anchors_per_order,
        batch_size=batch_size,
        max_padded_tokens=max_padded_tokens,
        output_root=output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
