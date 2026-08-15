"""Stage 3: build and encode the layer-2 dictionary knowledge bank on Modal.

This stage consumes the finalized, source-grounded definition manifest, links
every definition to its ordered constituent words, and extracts one Qwen
hidden-state vector per sense:

    definition text + fixed readout marker
    -> output of donor block 1 / hidden_states[2]
    -> final non-padding marker token
    -> one BF16 layer02 row

The completed Stage 2 bigram/trigram key files are used to report dictionary
coverage and retain unresolved corpus words for later dictionary supplements.

Full build:
    modal run modal_extract_definition_states.py

Resume:
    Run the same command again. Valid completed shards are reused.

Required finalized definition source manifest:
    /data/dendritron-stage3-definition-sources/canonical/
      definition_sources_manifest.json

Explicit rebuilds:
    --rebuild-inventory --inventory-only rewrites only the CPU inventory.
    --rebuild-bank rewrites only the selected smoke/full vector bank.

Outputs:
    /data/dendritron-stage3-dictionary/inventory/words.jsonl
    /data/dendritron-stage3-dictionary/inventory/senses.jsonl
    /data/dendritron-stage3-dictionary/inventory/dictionary.sqlite3
    /data/dendritron-stage3-dictionary/bank/shards/shard-00000.safetensors
    /data/dendritron-stage3-dictionary/bank/metadata/shard-00000.jsonl
    /data/dendritron-stage3-dictionary/bank/lookup.sqlite3
    /data/dendritron-stage3-dictionary/bank/manifest.json
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sqlite3
from argparse import Namespace
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

from modal_extract_states import (
    CORPUS_VOLUME_NAME,
    DATA_MOUNT,
    EXPECTED_FULL_TEXT_BLOCKS,
    EXPECTED_HIDDEN_SIZE,
    MODEL_CACHE,
    MODEL_ID,
    MODEL_MOUNT,
    MODEL_REVISION,
    MODEL_VOLUME_NAME,
    file_sha256,
)


APP_NAME = "dendritron-definition-extraction"
OUTPUT_ROOT = DATA_MOUNT / "dendritron-stage3-dictionary"
INVENTORY_ROOT = OUTPUT_ROOT / "inventory"
BANK_ROOT = OUTPUT_ROOT / "bank"
SMOKE_BANK_ROOT = OUTPUT_ROOT / "bank-smoke"

BIGRAM_KEYS = DATA_MOUNT / "dendritron-stage2/bigrams/keys.jsonl"
TRIGRAM_KEYS = DATA_MOUNT / "dendritron-stage2/trigrams/keys.jsonl"

HIDDEN_STATE_LAYER = 2
BLOCK_INDEX = HIDDEN_STATE_LAYER - 1
NUM_TEXT_BLOCKS_TO_LOAD = HIDDEN_STATE_LAYER
STORAGE_DTYPE = "bfloat16"
DEFAULT_SHARD_SIZE = 50_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_PADDED_TOKENS = 16_384
SMOKE_ROWS = 256
DEFAULT_DEFINITION_SOURCE_MANIFEST = (
    DATA_MOUNT
    / "dendritron-stage3-definition-sources/canonical/"
    "definition_sources_manifest.json"
)


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


@app.function(
    image=image,
    cpu=4.0,
    memory=16_384,
    timeout=6 * 60 * 60,
    volumes={str(MODEL_MOUNT): model_volume},
)
def prepare_model_assets() -> dict[str, Any]:
    """Resolve the same cached donor snapshot used by Stage 2."""
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE),
        max_workers=8,
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.model",
            "*.tiktoken",
            "*.txt",
            "*.py",
            "*.jinja",
        ],
        ignore_patterns=["*.bin", "*.pt", "*.gguf", "*.onnx"],
    )
    snapshot = Path(snapshot_path)
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Model snapshot is incomplete: {snapshot}")
    model_volume.commit()
    return {
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": snapshot.name,
        "snapshot_path": str(snapshot),
    }


@app.function(
    image=image,
    cpu=8.0,
    memory=65_536,
    timeout=6 * 60 * 60,
    volumes={str(DATA_MOUNT): corpus_volume},
)
def prepare_inventory(
    rebuild: bool = False,
    definition_source_manifest: str = str(DEFAULT_DEFINITION_SOURCE_MANIFEST),
) -> dict[str, Any]:
    """Build the finalized source sense inventory and definition-word graph."""
    from stage3_dictionary.build_dictionary_inventory import build_inventory

    source_manifest_path = Path(definition_source_manifest)
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            "Build and review the definition sources before the dictionary "
            f"inventory: {source_manifest_path}"
        )
    source_manifest_sha256 = file_sha256(source_manifest_path)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    coverage = source_manifest.get("coverage", {})
    if int(coverage.get("missing_ngram_words", -1)) != 0:
        raise RuntimeError(
            "The dictionary source contract must cover every Stage-2 Engram "
            "word before inventory or Qwen extraction. Review "
            f"{source_manifest_path.parent / 'coverage_report.json'} and add "
            "the remaining source-grounded definitions."
        )
    manifest_path = INVENTORY_ROOT / "inventory_manifest.json"
    if manifest_path.is_file() and not rebuild:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_source_manifest = manifest.get("definition_source_manifest")
        if (
            not recorded_source_manifest
            or recorded_source_manifest.get("path") != str(source_manifest_path)
            or recorded_source_manifest.get("sha256") != source_manifest_sha256
        ):
            raise RuntimeError(
                "The finalized definition source manifest differs from the "
                "existing inventory. The inventory and vector bank were "
                "preserved."
            )
        for artifact in manifest.get("input_definition_artifacts", []):
            source_path = Path(artifact["path"])
            if (
                not source_path.is_file()
                or file_sha256(source_path) != artifact["sha256"]
            ):
                raise RuntimeError(
                    f"Definition source changed since inventory creation: "
                    f"{source_path}. Existing outputs were preserved."
                )
        for artifact in manifest.get("input_ngram_key_artifacts", []):
            source_path = Path(artifact["path"])
            if (
                not source_path.is_file()
                or file_sha256(source_path) != artifact["sha256"]
            ):
                raise RuntimeError(
                    f"Stage 2 key source changed since inventory creation: "
                    f"{source_path}. Existing outputs were preserved."
                )
        required = (
            INVENTORY_ROOT / "words.jsonl",
            INVENTORY_ROOT / "senses.jsonl",
            INVENTORY_ROOT / "dictionary.sqlite3",
        )
        if not all(path.is_file() for path in required):
            raise RuntimeError(
                "The inventory manifest exists but one or more inventory "
                "artifacts are missing. Existing outputs were preserved."
            )
        for artifact in manifest.get("artifacts", {}).values():
            artifact_path = Path(artifact["path"])
            if (
                not artifact_path.is_file()
                or file_sha256(artifact_path) != artifact["sha256"]
            ):
                raise RuntimeError(
                    f"Inventory artifact failed its recorded hash: "
                    f"{artifact_path}. Existing outputs were preserved."
                )
        return manifest

    if rebuild and INVENTORY_ROOT.exists():
        shutil.rmtree(INVENTORY_ROOT)
    for path in (BIGRAM_KEYS, TRIGRAM_KEYS):
        if not path.is_file():
            raise FileNotFoundError(
                f"Completed Stage 2 key file is required: {path}"
            )
    report = build_inventory(
        Namespace(
            definition_manifest=source_manifest_path,
            wordnet=False,
            download_wordnet=False,
            definitions=[],
            ngram_keys=[BIGRAM_KEYS, TRIGRAM_KEYS],
            output=INVENTORY_ROOT,
        )
    )
    corpus_volume.commit()
    return report


def find_text_layers(model: Any) -> tuple[str, Any]:
    import torch

    for path in (
        "language_model.layers",
        "model.language_model.layers",
        "model.text_model.layers",
        "language_model.model.layers",
        "model.layers",
        "layers",
    ):
        value = model
        try:
            for component in path.split("."):
                value = getattr(value, component)
        except AttributeError:
            continue
        if isinstance(value, torch.nn.ModuleList) and value:
            return path, value
    raise RuntimeError("Could not locate donor text decoder blocks")


def truncate_definition_config(config: Any) -> dict[str, Any]:
    text_config = getattr(config, "text_config", config)
    full_count = int(getattr(text_config, "num_hidden_layers", 0))
    hidden_size = int(getattr(text_config, "hidden_size", 0))
    if full_count != EXPECTED_FULL_TEXT_BLOCKS:
        raise ValueError(
            f"Expected {EXPECTED_FULL_TEXT_BLOCKS} donor blocks, found {full_count}"
        )
    if hidden_size != EXPECTED_HIDDEN_SIZE:
        raise ValueError(
            f"Expected hidden size {EXPECTED_HIDDEN_SIZE}, found {hidden_size}"
        )

    trimmed_fields: list[str] = []
    for field, value in list(vars(text_config).items()):
        if isinstance(value, (list, tuple)) and len(value) == full_count:
            setattr(text_config, field, list(value)[:NUM_TEXT_BLOCKS_TO_LOAD])
            trimmed_fields.append(field)
    text_config.num_hidden_layers = NUM_TEXT_BLOCKS_TO_LOAD
    text_config.use_cache = False
    if hasattr(config, "text_config"):
        config.text_config = text_config
    return {
        "full_text_blocks": full_count,
        "loaded_text_blocks": NUM_TEXT_BLOCKS_TO_LOAD,
        "hidden_size": hidden_size,
        "trimmed_config_fields": trimmed_fields,
    }


def load_definition_donor(snapshot_path: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Donor tokenizer needs PAD or EOS")
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    details = truncate_definition_config(config)
    model = AutoModel.from_pretrained(
        snapshot_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="sdpa",
    )
    model.eval()
    layer_path, layers = find_text_layers(model)
    if len(layers) != NUM_TEXT_BLOCKS_TO_LOAD:
        raise RuntimeError(
            f"Expected {NUM_TEXT_BLOCKS_TO_LOAD} loaded blocks, found {len(layers)}"
        )
    return model, tokenizer, {
        **details,
        "captured_hidden_state_index": HIDDEN_STATE_LAYER,
        "captured_block_zero_based": BLOCK_INDEX,
        "captured_block_type": type(layers[BLOCK_INDEX]).__name__,
        "layer_module_path": layer_path,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
    }


class Layer02Tap:
    def __init__(self, block: Any) -> None:
        self.value: Any | None = None
        self.handle = block.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.value = output[0] if isinstance(output, tuple) else output

    def clear(self) -> None:
        self.value = None

    def close(self) -> None:
        self.clear()
        self.handle.remove()


def iter_senses(path: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        count = 0
        for raw_line in handle:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            required = {
                "sense_row",
                "sense_id",
                "word_id",
                "definition",
                "definition_words",
                "donor_text",
            }
            missing = required - set(record)
            if missing:
                raise KeyError(f"Sense row is missing fields: {sorted(missing)}")
            if int(record["sense_row"]) != count:
                raise ValueError(
                    f"Sense rows must be contiguous; expected {count}, "
                    f"found {record['sense_row']}"
                )
            yield record
            count += 1
            if limit is not None and count >= limit:
                return


def tokenizer_lengths(tokenizer: Any, texts: Sequence[str]) -> list[int]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    lengths = [len(values) for values in encoded]
    if any(length < 1 for length in lengths):
        raise ValueError("Definition donor input produced an empty token sequence")
    return lengths


def batched_senses(
    rows: Iterator[dict[str, Any]],
    tokenizer: Any,
    *,
    batch_size: int,
    max_padded_tokens: int,
) -> Iterator[list[dict[str, Any]]]:
    pending: list[dict[str, Any]] = []
    pending_lengths: list[int] = []
    for row in rows:
        length = tokenizer_lengths(tokenizer, [str(row["donor_text"])])[0]
        proposed_max = max(pending_lengths + [length])
        proposed_size = len(pending) + 1
        if pending and (
            proposed_size > batch_size
            or proposed_size * proposed_max > max_padded_tokens
        ):
            yield pending
            pending = []
            pending_lengths = []
        pending.append(row)
        pending_lengths.append(length)
    if pending:
        yield pending


def encode_layer02(
    rows: Sequence[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    tap: Layer02Tap,
) -> Any:
    import torch

    texts = [str(row["donor_text"]) for row in rows]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    encoded = {
        name: tensor.to("cuda:0", non_blocking=True) for name, tensor in encoded.items()
    }
    tap.clear()
    with torch.inference_mode():
        model(**encoded, use_cache=False)
    if tap.value is None:
        raise RuntimeError("Layer-2 donor hook did not fire")
    lengths = encoded["attention_mask"].sum(dim=1)
    batch_rows = torch.arange(lengths.shape[0], device=lengths.device)
    result = tap.value[batch_rows, lengths - 1].to(
        device="cpu",
        dtype=torch.bfloat16,
    )
    tap.clear()
    return result


def shard_paths(root: Path, shard_index: int) -> tuple[Path, Path]:
    return (
        root / "shards" / f"shard-{shard_index:05d}.safetensors",
        root / "metadata" / f"shard-{shard_index:05d}.jsonl",
    )


def valid_shard(
    tensor_path: Path,
    metadata_path: Path,
    expected_rows: int,
    expected_start_row: int = 0,
) -> bool:
    if not tensor_path.is_file() or not metadata_path.is_file():
        return False
    try:
        from safetensors import safe_open

        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"layer02"}:
                return False
            shape = tuple(handle.get_slice("layer02").get_shape())
            if shape != (expected_rows, EXPECTED_HIDDEN_SIZE):
                return False
        row_count = 0
        with metadata_path.open(encoding="utf-8") as handle:
            for local_row, raw_line in enumerate(handle):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                if int(record["sense_row"]) != expected_start_row + row_count:
                    return False
                if int(record["local_row"]) != row_count:
                    return False
                row_count += 1
        return row_count == expected_rows
    except Exception:
        return False


def extraction_source_contract(
    *,
    assets: dict[str, Any],
    inventory: dict[str, Any],
    senses_path: Path,
    smoke: bool,
    total_rows: int,
    shard_size: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": assets["resolved_revision"],
        "inventory_senses_sha256": file_sha256(senses_path),
        "definition_readout_marker": inventory["definition_readout_marker"],
        "hidden_state_layer": HIDDEN_STATE_LAYER,
        "pooling": "final_non_padding_marker_token",
        "storage_dtype": STORAGE_DTYPE,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "smoke": smoke,
        "rows": total_rows,
        "shard_size": shard_size,
    }


def completed_bank_is_valid(
    root: Path,
    *,
    contract: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_contract") != contract:
            return None
        total_rows = int(contract["rows"])
        shard_size = int(contract["shard_size"])
        shard_count = (total_rows + shard_size - 1) // shard_size
        for shard_index in range(shard_count):
            expected_rows = min(
                shard_size,
                total_rows - shard_index * shard_size,
            )
            tensor_path, metadata_path = shard_paths(root, shard_index)
            if not valid_shard(
                tensor_path,
                metadata_path,
                expected_rows,
                expected_start_row=shard_index * shard_size,
            ):
                return None
        lookup = Path(manifest["lookup"]["path"])
        if not lookup.is_file() or file_sha256(lookup) != manifest["lookup"]["sha256"]:
            return None
        return manifest
    except Exception:
        return None


def write_shard(
    root: Path,
    shard_index: int,
    rows: Sequence[dict[str, Any]],
    vectors: Any,
) -> None:
    from safetensors.torch import save_file

    tensor_path, metadata_path = shard_paths(root, shard_index)
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_tmp = tensor_path.with_name(tensor_path.name + ".tmp")
    metadata_tmp = metadata_path.with_name(metadata_path.name + ".tmp")
    save_file({"layer02": vectors.contiguous()}, str(tensor_tmp))
    with metadata_tmp.open("w", encoding="utf-8") as handle:
        for local_row, record in enumerate(rows):
            metadata = {
                key: value
                for key, value in record.items()
                if key != "donor_text"
            }
            metadata["shard_index"] = shard_index
            metadata["local_row"] = local_row
            handle.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(tensor_tmp, tensor_path)
    os.replace(metadata_tmp, metadata_path)


def build_lookup(root: Path, metadata_paths: Sequence[Path]) -> Path:
    output = root / "lookup.sqlite3"
    temporary = root / "lookup.sqlite3.tmp"
    if temporary.exists():
        temporary.unlink()
    connection = sqlite3.connect(str(temporary))
    try:
        connection.executescript(
            """
            CREATE TABLE senses (
                sense_row INTEGER PRIMARY KEY,
                sense_id TEXT NOT NULL UNIQUE,
                word_id INTEGER NOT NULL,
                surface TEXT NOT NULL,
                normalized TEXT NOT NULL,
                part_of_speech TEXT NOT NULL,
                definition TEXT NOT NULL,
                source TEXT NOT NULL,
                shard_index INTEGER NOT NULL,
                local_row INTEGER NOT NULL
            );
            CREATE INDEX senses_by_word ON senses(word_id);
            CREATE INDEX senses_by_normalized ON senses(normalized);
            """
        )
        expected = 0
        with connection:
            for path in metadata_paths:
                with path.open(encoding="utf-8") as handle:
                    for raw_line in handle:
                        record = json.loads(raw_line)
                        if int(record["sense_row"]) != expected:
                            raise ValueError(
                                f"Expected sense row {expected}, "
                                f"found {record['sense_row']}"
                            )
                        connection.execute(
                            """
                            INSERT INTO senses(
                                sense_row,
                                sense_id,
                                word_id,
                                surface,
                                normalized,
                                part_of_speech,
                                definition,
                                source,
                                shard_index,
                                local_row
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                expected,
                                record["sense_id"],
                                int(record["word_id"]),
                                record["surface"],
                                record["normalized"],
                                record["part_of_speech"],
                                record["definition"],
                                record["source"],
                                int(record["shard_index"]),
                                int(record["local_row"]),
                            ),
                        )
                        expected += 1
        connection.execute("PRAGMA optimize")
    finally:
        connection.close()
    os.replace(temporary, output)
    return output


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
def extract_definition_bank(
    assets: dict[str, Any],
    inventory: dict[str, Any],
    *,
    smoke: bool = False,
    rebuild: bool = False,
    shard_size: int = DEFAULT_SHARD_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
) -> dict[str, Any]:
    import torch

    root = SMOKE_BANK_ROOT if smoke else BANK_ROOT
    if rebuild and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    senses_path = INVENTORY_ROOT / "senses.jsonl"
    total_inventory_rows = int(inventory["sense_rows"])
    total_rows = min(total_inventory_rows, SMOKE_ROWS) if smoke else total_inventory_rows
    source_contract = extraction_source_contract(
        assets=assets,
        inventory=inventory,
        senses_path=senses_path,
        smoke=smoke,
        total_rows=total_rows,
        shard_size=shard_size,
    )
    contract_path = root / "source_contract.json"
    if contract_path.is_file():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing_contract != source_contract:
            raise RuntimeError(
                "The existing definition-bank shards belong to a different "
                "inventory/model contract. They were preserved. Use "
                "--rebuild-bank only after confirming the source change."
            )
    elif any((root / "shards").glob("shard-*.safetensors")):
        raise RuntimeError(
            "Existing vector shards have no source contract, so safe resume "
            "cannot prove their identity. They were preserved. Inspect them "
            "before choosing --rebuild-bank."
        )
    else:
        atomic_write_text(
            contract_path,
            json.dumps(source_contract, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )
        corpus_volume.commit()

    completed = completed_bank_is_valid(root, contract=source_contract)
    if completed is not None:
        return completed

    model, tokenizer, model_details = load_definition_donor(assets["snapshot_path"])
    _, layers = find_text_layers(model)
    tap = Layer02Tap(layers[BLOCK_INDEX])

    all_rows = iter_senses(senses_path, limit=total_rows)
    shard_index = 0
    global_row = 0
    metadata_paths: list[Path] = []
    try:
        while global_row < total_rows:
            expected_rows = min(shard_size, total_rows - global_row)
            tensor_path, metadata_path = shard_paths(root, shard_index)
            if valid_shard(
                tensor_path,
                metadata_path,
                expected_rows,
                expected_start_row=global_row,
            ):
                for _ in range(expected_rows):
                    next(all_rows)
                metadata_paths.append(metadata_path)
                global_row += expected_rows
                shard_index += 1
                continue

            tensor_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            rows: list[dict[str, Any]] = []
            while len(rows) < expected_rows:
                rows.append(next(all_rows))
            vector_batches = []
            for batch in batched_senses(
                iter(rows),
                tokenizer,
                batch_size=batch_size,
                max_padded_tokens=max_padded_tokens,
            ):
                vector_batches.append(
                    encode_layer02(batch, model, tokenizer, tap)
                )
            vectors = torch.cat(vector_batches, dim=0)
            if tuple(vectors.shape) != (expected_rows, EXPECTED_HIDDEN_SIZE):
                raise RuntimeError(
                    f"Bad shard shape {tuple(vectors.shape)} at {shard_index}"
                )
            write_shard(root, shard_index, rows, vectors)
            if not valid_shard(
                tensor_path,
                metadata_path,
                expected_rows,
                expected_start_row=global_row,
            ):
                raise RuntimeError(f"Shard {shard_index} failed validation")
            metadata_paths.append(metadata_path)
            corpus_volume.commit()
            global_row += expected_rows
            shard_index += 1
            del rows, vectors, vector_batches
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        tap.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    lookup_path = build_lookup(root, metadata_paths)
    manifest = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "source_contract": source_contract,
        "smoke": smoke,
        "source_inventory_manifest": inventory,
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": assets["resolved_revision"],
        "definition_readout_marker": inventory["definition_readout_marker"],
        "hidden_state_layer": HIDDEN_STATE_LAYER,
        "block_index_zero_based": BLOCK_INDEX,
        "pooling": "final_non_padding_marker_token",
        "storage_dtype": STORAGE_DTYPE,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "rows": total_rows,
        "shard_size": shard_size,
        "shards": len(metadata_paths),
        "model_details": model_details,
        "lookup": {
            "path": str(lookup_path),
            "sha256": file_sha256(lookup_path),
        },
        "inventory_senses_sha256": file_sha256(senses_path),
    }
    manifest_path = root / "manifest.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    corpus_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    smoke: bool = False,
    rebuild_inventory: bool = False,
    rebuild_bank: bool = False,
    inventory_only: bool = False,
    definition_source_manifest: str = str(DEFAULT_DEFINITION_SOURCE_MANIFEST),
    shard_size: int = DEFAULT_SHARD_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
) -> None:
    inventory = prepare_inventory.remote(
        rebuild=rebuild_inventory,
        definition_source_manifest=definition_source_manifest,
    )
    if inventory_only:
        print(json.dumps(inventory, ensure_ascii=False, indent=2))
        return
    assets = prepare_model_assets.remote()
    result = extract_definition_bank.remote(
        assets,
        inventory,
        smoke=smoke,
        rebuild=rebuild_bank,
        shard_size=shard_size,
        batch_size=batch_size,
        max_padded_tokens=max_padded_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
