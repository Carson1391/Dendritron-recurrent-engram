"""
Stage 2: extract frozen donor hidden-state banks on Modal.

This consumes the Stage 1 text inventories directly:

    /data/dendritron-stage1/final/top_bigrams.jsonl
    /data/dendritron-stage1/final/top_trigrams.jsonl

For every text key, it:

1. Tokenizes the phrase with the Qwen donor tokenizer.
2. Runs that phrase by itself through the donor.
3. Captures hidden_state[8] and hidden_state[24], meaning the outputs of
   decoder blocks 7 and 23 in zero-based ModuleList indexing.
4. Takes the final non-padding token from each captured state.
5. Writes a separate 2-gram table and 3-gram table. Each Engram row contains
   both BF16 donor views: layer08 and layer24.

Canonical row order inside each table:

    bigrams/ rows 0..499,999 = Stage 1 bigrams in rank order
    trigrams/ rows 0..499,999 = Stage 1 trigrams in rank order

The persistent key is UTF-8 text. Donor token IDs are transient inputs to the
forward pass; recipient token IDs or hash addresses can be derived later.

The donor is loaded with AutoModel, so no language-model output head is
allocated. Its text stack is truncated to the first 24 blocks before weights
are loaded because blocks 24..39 cannot affect states captured at blocks 7
and 23.

First run a small end-to-end check:

    modal run modal_extract_states.py --smoke

Then run all one million keys:

    modal run modal_extract_states.py

Resume:

    Run the same command again. Completed shards are validated and reused.

Clean rebuild:

    modal run modal_extract_states.py --force

Outputs on the "dendritron-corpus" Modal Volume:

    /data/dendritron-stage2/manifest.json
    /data/dendritron-stage2/bigrams/keys.jsonl
    /data/dendritron-stage2/bigrams/lookup.sqlite3
    /data/dendritron-stage2/bigrams/shards/shard-00000.safetensors
    /data/dendritron-stage2/bigrams/metadata/shard-00000.jsonl
    /data/dendritron-stage2/trigrams/keys.jsonl
    /data/dendritron-stage2/trigrams/lookup.sqlite3
    /data/dendritron-stage2/trigrams/shards/shard-00000.safetensors
    /data/dendritron-stage2/trigrams/metadata/shard-00000.jsonl

Each Safetensors shard has two tensor keys, ``layer08`` and ``layer24``.

The initial model download goes straight to a separate Modal Volume. It does
not download the model to the computer launching this job.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Locked extraction specification
# ---------------------------------------------------------------------------

APP_NAME = "dendritron-donor-extraction"

CORPUS_VOLUME_NAME = "dendritron-corpus"
MODEL_VOLUME_NAME = "dendritron-donor-model"

DATA_MOUNT = Path("/data")
MODEL_MOUNT = Path("/models")

BIGRAM_PATH = DATA_MOUNT / "dendritron-stage1/final/top_bigrams.jsonl"
TRIGRAM_PATH = DATA_MOUNT / "dendritron-stage1/final/top_trigrams.jsonl"
CORPUS_STATS_PATH = DATA_MOUNT / "dendritron-stage1/final/corpus_statistics.json"

OUTPUT_ROOT = DATA_MOUNT / "dendritron-stage2"
SMOKE_OUTPUT_ROOT = DATA_MOUNT / "dendritron-stage2-smoke"

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "main"
MODEL_CACHE = MODEL_MOUNT / "huggingface"

# Hugging Face hidden_states[0] is the embedding output. Therefore
# hidden_states[8] is the output of decoder block 7, and hidden_states[24] is
# the output of decoder block 23.
HIDDEN_STATE_LAYERS = (8, 24)
BLOCK_INDICES = tuple(layer - 1 for layer in HIDDEN_STATE_LAYERS)
NUM_TEXT_BLOCKS_TO_LOAD = max(BLOCK_INDICES) + 1
EXPECTED_FULL_TEXT_BLOCKS = 40
EXPECTED_HIDDEN_SIZE = 2048

EXPECTED_ROWS_PER_ORDER = 500_000
DEFAULT_SHARD_SIZE = 50_000
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_PADDED_TOKENS = 4_096
SMOKE_ROWS_PER_ORDER = 128

STORAGE_DTYPE = "bfloat16"
POOLING = "final_non_pad_token"
ADD_SPECIAL_TOKENS = False
PADDING_SIDE = "right"


# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

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
            # These strings are evaluated by the local launcher. Keep the
            # container's POSIX mount paths when launching from Windows.
            "HF_HOME": "/models/huggingface",
            "HF_HUB_CACHE": "/models/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "true",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


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


def count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixed_output_root(smoke: bool) -> Path:
    return SMOKE_OUTPUT_ROOT if smoke else OUTPUT_ROOT


def expected_rows_per_order(smoke: bool) -> int:
    return SMOKE_ROWS_PER_ORDER if smoke else EXPECTED_ROWS_PER_ORDER


def source_contract(smoke: bool) -> list[dict[str, Any]]:
    rows = expected_rows_per_order(smoke)
    return [
        {"n": 2, "path": str(BIGRAM_PATH), "expected_rows": rows},
        {"n": 3, "path": str(TRIGRAM_PATH), "expected_rows": rows},
    ]


def validate_source_file(path: Path, expected_n: int, expected_rows: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage 1 input: {path}")

    seen = 0
    with path.open(encoding="utf-8") as handle:
        for source_index, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            if int(row.get("n", -1)) != expected_n:
                raise ValueError(
                    f"{path}:{source_index + 1}: expected n={expected_n}, "
                    f"found {row.get('n')!r}"
                )
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"{path}:{source_index + 1}: empty text key")
            if int(row.get("frequency", -1)) < 1:
                raise ValueError(
                    f"{path}:{source_index + 1}: invalid frequency "
                    f"{row.get('frequency')!r}"
                )
            if int(row.get("rank", seen + 1)) != seen + 1:
                raise ValueError(
                    f"{path}:{source_index + 1}: expected rank {seen + 1}, "
                    f"found {row.get('rank')!r}"
                )
            seen += 1
            if seen == expected_rows:
                break

    if seen != expected_rows:
        raise ValueError(
            f"{path}: expected at least {expected_rows:,} valid rows, found {seen:,}"
        )


def build_input_info(smoke: bool) -> dict[str, Any]:
    contracts = source_contract(smoke)
    for source in contracts:
        validate_source_file(
            Path(source["path"]),
            expected_n=int(source["n"]),
            expected_rows=int(source["expected_rows"]),
        )

    files = {
        str(BIGRAM_PATH): {
            "sha256": file_sha256(BIGRAM_PATH),
            "bytes": BIGRAM_PATH.stat().st_size,
            "rows_selected": contracts[0]["expected_rows"],
        },
        str(TRIGRAM_PATH): {
            "sha256": file_sha256(TRIGRAM_PATH),
            "bytes": TRIGRAM_PATH.stat().st_size,
            "rows_selected": contracts[1]["expected_rows"],
        },
    }
    info: dict[str, Any] = {
        "smoke": smoke,
        "sources": contracts,
        "files": files,
        "total_rows": sum(int(item["expected_rows"]) for item in contracts),
    }
    if CORPUS_STATS_PATH.is_file():
        info["corpus_statistics_sha256"] = file_sha256(CORPUS_STATS_PATH)
    info["fingerprint"] = json_fingerprint(info)
    return info


def iter_source_rows(
    path: Path,
    expected_n: int,
    take: int,
) -> Iterator[dict[str, Any]]:
    yielded = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            source = json.loads(raw_line)
            yield {
                "text": source["text"],
                "frequency": int(source["frequency"]),
                "n": expected_n,
                "source_rank": int(source.get("rank", yielded + 1)),
            }
            yielded += 1
            if yielded == take:
                return
    raise ValueError(f"{path}: ended after {yielded:,} rows; expected {take:,}")


def iter_table_rows(source: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for row_index, row in enumerate(
        iter_source_rows(
            Path(source["path"]),
            expected_n=int(source["n"]),
            take=int(source["expected_rows"]),
        )
    ):
        row["row_index"] = row_index
        yield row


def chunked_rows(
    rows: Iterable[dict[str, Any]],
    chunk_size: int,
) -> Iterator[list[dict[str, Any]]]:
    chunk: list[dict[str, Any]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# ---------------------------------------------------------------------------
# CPU preflight and model staging
# ---------------------------------------------------------------------------


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
def prepare_assets(smoke: bool = False) -> dict[str, Any]:
    """Validate Stage 1 and cache the donor directly on a Modal Volume."""
    from huggingface_hub import snapshot_download

    input_info = build_input_info(smoke)
    print(
        f"[input] validated {input_info['total_rows']:,} selected rows; "
        f"fingerprint={input_info['fingerprint'][:12]}"
    )

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
    commit_hash = snapshot.name
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"Model snapshot is incomplete: {snapshot}")

    model_volume.commit()
    return {
        "input": input_info,
        "model_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": commit_hash,
        "snapshot_path": str(snapshot),
    }


# ---------------------------------------------------------------------------
# Model and layer plumbing
# ---------------------------------------------------------------------------


def find_text_layers(model: Any) -> tuple[str, Any]:
    import torch

    candidates = (
        "language_model.layers",
        "model.language_model.layers",
        "model.text_model.layers",
        "language_model.model.layers",
        "model.layers",
        "layers",
    )
    for path in candidates:
        value = model
        try:
            for component in path.split("."):
                value = getattr(value, component)
        except AttributeError:
            continue
        if isinstance(value, torch.nn.ModuleList) and value:
            return path, value
    raise RuntimeError("Could not locate the donor text decoder blocks")


def truncate_text_config(config: Any) -> dict[str, Any]:
    text_config = getattr(config, "text_config", config)
    full_count = int(getattr(text_config, "num_hidden_layers", 0))
    hidden_size = int(getattr(text_config, "hidden_size", 0))
    if full_count < NUM_TEXT_BLOCKS_TO_LOAD:
        raise ValueError(
            f"Donor exposes {full_count} text blocks; "
            f"{NUM_TEXT_BLOCKS_TO_LOAD} are required"
        )
    if hidden_size != EXPECTED_HIDDEN_SIZE:
        raise ValueError(
            f"Expected hidden_size={EXPECTED_HIDDEN_SIZE}, found {hidden_size}"
        )

    original_layer_types = list(getattr(text_config, "layer_types", []))
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
        "original_layer_types": original_layer_types,
    }


def load_donor(snapshot_path: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    torch.set_float32_matmul_precision("high")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    tokenizer.padding_side = PADDING_SIDE
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Donor tokenizer has neither a PAD nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        snapshot_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    truncation = truncate_text_config(config)

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

    if hasattr(model, "lm_head"):
        raise RuntimeError(
            "AutoModel unexpectedly returned a language-model head; "
            "refusing to waste GPU memory"
        )

    layer_path, layers = find_text_layers(model)
    if len(layers) != NUM_TEXT_BLOCKS_TO_LOAD:
        raise RuntimeError(
            f"Expected {NUM_TEXT_BLOCKS_TO_LOAD} loaded text blocks at "
            f"{layer_path}, found {len(layers)}"
        )

    layer_types = list(getattr(config.text_config, "layer_types", []))
    block_types = {
        str(block): layer_types[block]
        if block < len(layer_types)
        else type(layers[block]).__name__
        for block in BLOCK_INDICES
    }
    details = {
        **truncation,
        "layer_module_path": layer_path,
        "captured_blocks_zero_based": list(BLOCK_INDICES),
        "captured_hidden_state_indices": list(HIDDEN_STATE_LAYERS),
        "captured_block_types": block_types,
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "lm_head_loaded": False,
    }
    return model, tokenizer, details


class LayerTap:
    """Capture only the two decoder-block outputs needed for the banks."""

    def __init__(self, layers: Sequence[Any]) -> None:
        self.buffer: dict[int, Any] = {}
        self.handles = [
            layers[block].register_forward_hook(self._make_hook(block))
            for block in BLOCK_INDICES
        ]

    def _make_hook(self, block: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self.buffer[block] = output[0] if isinstance(output, tuple) else output

        return hook

    def clear(self) -> None:
        self.buffer.clear()

    def close(self) -> None:
        self.clear()
        for handle in self.handles:
            handle.remove()
        self.handles = []


def tokenizer_lengths(tokenizer: Any, texts: list[str]) -> list[int]:
    lengths: list[int] = []
    step = 4_096
    for start in range(0, len(texts), step):
        encoded = tokenizer(
            texts[start : start + step],
            add_special_tokens=ADD_SPECIAL_TOKENS,
            padding=False,
            truncation=False,
        )["input_ids"]
        lengths.extend(len(token_ids) for token_ids in encoded)
    if any(length < 1 for length in lengths):
        raise ValueError("The donor tokenizer produced an empty token sequence")
    return lengths


def encode_batch(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    tap: LayerTap,
) -> dict[int, Any]:
    import torch

    encoded = tokenizer(
        texts,
        add_special_tokens=ADD_SPECIAL_TOKENS,
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

    missing = set(BLOCK_INDICES) - set(tap.buffer)
    if missing:
        raise RuntimeError(f"Layer hooks did not fire for blocks {sorted(missing)}")

    lengths = encoded["attention_mask"].sum(dim=1)
    batch_rows = torch.arange(lengths.shape[0], device=lengths.device)
    last_positions = lengths - 1

    result: dict[int, Any] = {}
    for block in BLOCK_INDICES:
        hidden = tap.buffer[block]
        result[block] = hidden[batch_rows, last_positions].to(
            device="cpu",
            dtype=torch.bfloat16,
        )

    tap.clear()
    del encoded
    return result


# ---------------------------------------------------------------------------
# Shard writing and validation
# ---------------------------------------------------------------------------


def table_name(n: int) -> str:
    if n == 2:
        return "bigrams"
    if n == 3:
        return "trigrams"
    raise ValueError(f"Unsupported n-gram order: {n}")


def table_root(output_root: Path, n: int) -> Path:
    return output_root / table_name(n)


def tensor_shard_path(output_root: Path, n: int, shard_index: int) -> Path:
    return (
        table_root(output_root, n) / "shards" / f"shard-{shard_index:05d}.safetensors"
    )


def metadata_shard_path(output_root: Path, n: int, shard_index: int) -> Path:
    return table_root(output_root, n) / "metadata" / f"shard-{shard_index:05d}.jsonl"


def validate_tensor_shard(path: Path, expected_rows: int) -> bool:
    if not path.is_file():
        return False
    try:
        from safetensors import SafetensorError, safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            expected_keys = [f"layer{layer:02d}" for layer in HIDDEN_STATE_LAYERS]
            if sorted(handle.keys()) != sorted(expected_keys):
                return False
            return all(
                tuple(handle.get_tensor(key).shape)
                == (expected_rows, EXPECTED_HIDDEN_SIZE)
                and str(handle.get_tensor(key).dtype) == "torch.bfloat16"
                for key in expected_keys
            )
    except (OSError, ValueError, KeyError, RuntimeError, SafetensorError):
        return False


def validate_metadata_shard(
    path: Path,
    expected_first_row: int,
    expected_rows: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        first: dict[str, Any] | None = None
        last: dict[str, Any] | None = None
        count = 0
        with path.open(encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                if first is None:
                    first = row
                last = row
                count += 1
        return (
            count == expected_rows
            and first is not None
            and last is not None
            and int(first["row_index"]) == expected_first_row
            and int(last["row_index"]) == expected_first_row + expected_rows - 1
        )
    except (OSError, ValueError, KeyError):
        return False


def shard_is_complete(
    output_root: Path,
    n: int,
    shard_index: int,
    expected_first_row: int,
    expected_rows: int,
) -> bool:
    tensors_ok = validate_tensor_shard(
        tensor_shard_path(output_root, n, shard_index),
        expected_rows,
    )
    metadata_ok = validate_metadata_shard(
        metadata_shard_path(output_root, n, shard_index),
        expected_first_row,
        expected_rows,
    )
    return tensors_ok and metadata_ok


def write_tensor_shard(
    path: Path,
    tensors: dict[str, Any],
    metadata: dict[str, str],
) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(
        {name: tensor.contiguous() for name, tensor in tensors.items()},
        temporary,
        metadata=metadata,
    )
    os.replace(temporary, path)


def write_metadata_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def remove_partial_shard(output_root: Path, n: int, shard_index: int) -> None:
    paths = [
        metadata_shard_path(output_root, n, shard_index),
        tensor_shard_path(output_root, n, shard_index),
    ]
    for path in paths:
        if path.exists():
            path.unlink()


def extract_shard(
    rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    tap: LayerTap,
    batch_size: int,
    max_padded_tokens: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    import torch

    texts = [row["text"] for row in rows]
    token_lengths = tokenizer_lengths(tokenizer, texts)
    order = sorted(range(len(rows)), key=token_lengths.__getitem__)

    banks = {
        block: torch.empty(
            (len(rows), EXPECTED_HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device="cpu",
        )
        for block in BLOCK_INDICES
    }

    cursor = 0
    oom_retries = 0
    successful_batches = 0
    while cursor < len(order):
        current_batch_size = min(batch_size, len(order) - cursor)

        while current_batch_size > 1:
            candidate = order[cursor : cursor + current_batch_size]
            padded_tokens = current_batch_size * token_lengths[candidate[-1]]
            if padded_tokens <= max_padded_tokens:
                break
            current_batch_size = max(1, current_batch_size // 2)

        while True:
            selected = order[cursor : cursor + current_batch_size]
            batch_texts = [texts[index] for index in selected]
            try:
                captured = encode_batch(batch_texts, model, tokenizer, tap)
                break
            except torch.OutOfMemoryError:
                tap.clear()
                gc.collect()
                torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                oom_retries += 1
                print(f"[oom] retrying with batch_size={current_batch_size}")

        destination = torch.tensor(selected, dtype=torch.long)
        for block in BLOCK_INDICES:
            banks[block].index_copy_(0, destination, captured[block])
        cursor += current_batch_size
        successful_batches += 1
        del captured, destination

    for row, token_count in zip(rows, token_lengths):
        row["donor_token_count"] = token_count

    stats = {
        "rows": len(rows),
        "batches": successful_batches,
        "oom_retries": oom_retries,
        "min_donor_tokens": min(token_lengths),
        "max_donor_tokens": max(token_lengths),
        "mean_donor_tokens": sum(token_lengths) / len(token_lengths),
    }
    return banks, stats


# ---------------------------------------------------------------------------
# Final key index
# ---------------------------------------------------------------------------


def ordered_metadata_paths(
    output_root: Path,
    n: int,
    shard_count: int,
) -> list[Path]:
    return [
        metadata_shard_path(output_root, n, shard_index)
        for shard_index in range(shard_count)
    ]


def build_key_artifacts(
    output_root: Path,
    n: int,
    shard_count: int,
    expected_total_rows: int,
) -> dict[str, Any]:
    destination = table_root(output_root, n)
    keys_path = destination / "keys.jsonl"
    keys_temporary = keys_path.with_name(keys_path.name + ".tmp")
    database_path = destination / "lookup.sqlite3"
    database_temporary = database_path.with_name(database_path.name + ".tmp")

    if keys_temporary.exists():
        keys_temporary.unlink()
    if database_temporary.exists():
        database_temporary.unlink()

    connection = sqlite3.connect(database_temporary)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE keys (
            text TEXT PRIMARY KEY,
            row_index INTEGER NOT NULL UNIQUE,
            n INTEGER NOT NULL,
            frequency INTEGER NOT NULL,
            source_rank INTEGER NOT NULL,
            donor_token_count INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX keys_by_row ON keys(row_index);
        """
    )

    total_rows = 0
    pending: list[tuple[Any, ...]] = []
    with keys_temporary.open("w", encoding="utf-8") as keys_handle:
        for path in ordered_metadata_paths(output_root, n, shard_count):
            with path.open(encoding="utf-8") as metadata_handle:
                for raw_line in metadata_handle:
                    if not raw_line.strip():
                        continue
                    row = json.loads(raw_line)
                    if int(row["row_index"]) != total_rows:
                        raise ValueError(
                            f"Non-canonical row sequence at {path}: "
                            f"expected {total_rows}, found {row['row_index']}"
                        )
                    if int(row["n"]) != n:
                        raise ValueError(f"{path}: expected n={n}, found n={row['n']}")
                    keys_handle.write(
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    )
                    keys_handle.write("\n")
                    pending.append(
                        (
                            row["text"],
                            int(row["row_index"]),
                            int(row["n"]),
                            int(row["frequency"]),
                            int(row["source_rank"]),
                            int(row["donor_token_count"]),
                        )
                    )
                    total_rows += 1
                    if len(pending) == 10_000:
                        connection.executemany(
                            "INSERT INTO keys VALUES (?, ?, ?, ?, ?, ?)",
                            pending,
                        )
                        pending.clear()
        if pending:
            connection.executemany(
                "INSERT INTO keys VALUES (?, ?, ?, ?, ?, ?)",
                pending,
            )

    if total_rows != expected_total_rows:
        connection.close()
        raise ValueError(
            f"Expected {expected_total_rows:,} final keys, found {total_rows:,}"
        )

    connection.commit()
    integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    connection.close()
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed: {integrity}")

    os.replace(keys_temporary, keys_path)
    os.replace(database_temporary, database_path)
    return {
        "keys_jsonl": str(keys_path),
        "lookup_sqlite": str(database_path),
        "rows": total_rows,
        "n": n,
        "keys_sha256": file_sha256(keys_path),
        "lookup_sha256": file_sha256(database_path),
    }


# ---------------------------------------------------------------------------
# GPU extraction
# ---------------------------------------------------------------------------


def extraction_spec(
    prepared: dict[str, Any],
    smoke: bool,
    shard_size: int,
) -> dict[str, Any]:
    value = {
        "schema_version": 2,
        "model_id": MODEL_ID,
        "model_revision": prepared["resolved_revision"],
        "input_fingerprint": prepared["input"]["fingerprint"],
        "smoke": smoke,
        "hidden_state_layers": list(HIDDEN_STATE_LAYERS),
        "block_indices_zero_based": list(BLOCK_INDICES),
        "num_text_blocks_loaded": NUM_TEXT_BLOCKS_TO_LOAD,
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "storage_dtype": STORAGE_DTYPE,
        "pooling": POOLING,
        "add_special_tokens": ADD_SPECIAL_TOKENS,
        "padding_side": PADDING_SIDE,
        "shard_size": shard_size,
        "key_type": "utf8_surface_text",
        "table_partition": "one_table_per_ngram_order",
        "row_order": "source_frequency_rank_within_each_table",
    }
    value["fingerprint"] = json_fingerprint(value)
    return value


def prepare_output_directory(
    output_root: Path,
    spec: dict[str, Any],
    force: bool,
) -> None:
    spec_path = output_root / "run_spec.json"
    if force and output_root.exists():
        shutil.rmtree(output_root)

    if spec_path.is_file():
        existing = load_json(spec_path)
        if existing.get("fingerprint") != spec["fingerprint"]:
            raise RuntimeError(
                f"{output_root} belongs to a different extraction spec. "
                "Run with --force to replace that fixed output directory."
            )
    elif output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(
            f"{output_root} contains data without a run_spec.json. "
            "Run with --force to replace that fixed output directory."
        )

    output_root.mkdir(parents=True, exist_ok=True)
    for n in (2, 3):
        (table_root(output_root, n) / "shards").mkdir(
            parents=True,
            exist_ok=True,
        )
        (table_root(output_root, n) / "metadata").mkdir(
            parents=True,
            exist_ok=True,
        )
    if not spec_path.exists():
        atomic_write_text(
            spec_path,
            json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
        )
        corpus_volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=8.0,
    memory=32_768,
    timeout=24 * 60 * 60,
    volumes={
        str(DATA_MOUNT): corpus_volume,
        str(MODEL_MOUNT): model_volume,
    },
)
def extract_banks(
    prepared: dict[str, Any],
    smoke: bool = False,
    force: bool = False,
    shard_size: int = DEFAULT_SHARD_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
) -> dict[str, Any]:
    import torch

    if shard_size < 1 or batch_size < 1 or max_padded_tokens < 1:
        raise ValueError(
            "shard_size, batch_size, and max_padded_tokens must be positive"
        )
    if prepared["model_id"] != MODEL_ID:
        raise ValueError("Prepared model does not match the locked donor")
    if not str(prepared["snapshot_path"]).startswith("/models/"):
        raise ValueError("Prepared model snapshot is outside the Modal model volume")

    current_input = build_input_info(smoke)
    if current_input["fingerprint"] != prepared["input"]["fingerprint"]:
        raise RuntimeError("Stage 1 inputs changed between preflight and GPU startup")

    output_root = fixed_output_root(smoke)
    spec = extraction_spec(prepared, smoke, shard_size)
    prepare_output_directory(output_root, spec, force)

    completed_manifest_path = output_root / "manifest.json"
    if completed_manifest_path.is_file():
        completed = load_json(completed_manifest_path)
        if (
            completed.get("completed") is True
            and completed.get("extraction_fingerprint") == spec["fingerprint"]
        ):
            print(f"[done] completed extraction already exists at {output_root}")
            return completed

    total_rows = int(current_input["total_rows"])
    table_shard_counts = {
        int(source["n"]): math.ceil(int(source["expected_rows"]) / shard_size)
        for source in current_input["sources"]
    }
    print(
        f"[run] {total_rows:,} keys across two Engram tables; "
        f"each row stores layers {HIDDEN_STATE_LAYERS}; output={output_root}"
    )
    print(
        "[keys] text remains canonical; donor token IDs are used only "
        "inside each forward pass"
    )

    model, tokenizer, model_details = load_donor(prepared["snapshot_path"])
    print(
        f"[model] {model_details['model_class']} with "
        f"{model_details['loaded_text_blocks']}/"
        f"{model_details['full_text_blocks']} text blocks; no LM head"
    )
    print(
        "[layers] "
        + ", ".join(
            f"hidden_state[{layer}] <- block[{block}] "
            f"({model_details['captured_block_types'][str(block)]})"
            for layer, block in zip(HIDDEN_STATE_LAYERS, BLOCK_INDICES)
        )
    )

    _, layers = find_text_layers(model)
    tap = LayerTap(layers)
    started = time.time()
    completed_rows = 0
    shard_stats: list[dict[str, Any]] = []

    try:
        for source in current_input["sources"]:
            n = int(source["n"])
            name = table_name(n)
            table_shard_count = table_shard_counts[n]
            rows = iter_table_rows(source)
            for shard_index, shard_rows in enumerate(chunked_rows(rows, shard_size)):
                first_row = int(shard_rows[0]["row_index"])
                row_count = len(shard_rows)
                if shard_is_complete(
                    output_root,
                    n,
                    shard_index,
                    expected_first_row=first_row,
                    expected_rows=row_count,
                ):
                    completed_rows += row_count
                    print(
                        f"[{name} shard {shard_index + 1}/"
                        f"{table_shard_count}] reused rows {first_row:,}.."
                        f"{first_row + row_count - 1:,}"
                    )
                    continue

                remove_partial_shard(output_root, n, shard_index)
                shard_started = time.time()
                banks, stats = extract_shard(
                    shard_rows,
                    model,
                    tokenizer,
                    tap,
                    batch_size=batch_size,
                    max_padded_tokens=max_padded_tokens,
                )

                write_tensor_shard(
                    tensor_shard_path(output_root, n, shard_index),
                    {
                        f"layer{layer:02d}": banks[block]
                        for layer, block in zip(
                            HIDDEN_STATE_LAYERS,
                            BLOCK_INDICES,
                        )
                    },
                    metadata={
                        "n": str(n),
                        "table": name,
                        "tensor_keys": ",".join(
                            f"layer{layer:02d}" for layer in HIDDEN_STATE_LAYERS
                        ),
                        "hidden_state_layers": ",".join(
                            str(layer) for layer in HIDDEN_STATE_LAYERS
                        ),
                        "decoder_blocks_zero_based": ",".join(
                            str(block) for block in BLOCK_INDICES
                        ),
                        "first_row": str(first_row),
                        "rows": str(row_count),
                        "hidden_size": str(EXPECTED_HIDDEN_SIZE),
                        "dtype": STORAGE_DTYPE,
                        "pooling": POOLING,
                        "model_id": MODEL_ID,
                        "model_revision": prepared["resolved_revision"],
                    },
                )
                write_metadata_shard(
                    metadata_shard_path(output_root, n, shard_index),
                    shard_rows,
                )

                del banks
                gc.collect()
                torch.cuda.empty_cache()
                corpus_volume.commit()

                completed_rows += row_count
                elapsed = time.time() - shard_started
                stats.update(
                    {
                        "n": n,
                        "table": name,
                        "shard_index": shard_index,
                        "first_row": first_row,
                        "seconds": elapsed,
                    }
                )
                shard_stats.append(stats)
                total_elapsed = time.time() - started
                print(
                    f"[{name} shard {shard_index + 1}/"
                    f"{table_shard_count}] wrote {row_count:,} rows in "
                    f"{elapsed:.1f}s; "
                    f"{completed_rows / max(total_elapsed, 1e-6):.1f} "
                    "rows/s this invocation"
                )
    finally:
        tap.close()

    table_artifacts: dict[str, Any] = {}
    for source in current_input["sources"]:
        n = int(source["n"])
        name = table_name(n)
        table_rows = int(source["expected_rows"])
        table_shard_count = table_shard_counts[n]
        keys = build_key_artifacts(
            output_root,
            n=n,
            shard_count=table_shard_count,
            expected_total_rows=table_rows,
        )
        entries: list[dict[str, Any]] = []
        for shard_index in range(table_shard_count):
            path = tensor_shard_path(output_root, n, shard_index)
            entries.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        table_artifacts[name] = {
            "n": n,
            "rows": table_rows,
            "shard_count": table_shard_count,
            "tensor_keys": [f"layer{layer:02d}" for layer in HIDDEN_STATE_LAYERS],
            "tensor_files": entries,
            **keys,
        }

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "completed": True,
        "completed_at_utc": utc_now(),
        "extraction_fingerprint": spec["fingerprint"],
        "model": {
            "id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_revision": prepared["resolved_revision"],
            **model_details,
        },
        "states": {
            "hidden_state_layers": list(HIDDEN_STATE_LAYERS),
            "decoder_blocks_zero_based": list(BLOCK_INDICES),
            "hidden_size": EXPECTED_HIDDEN_SIZE,
            "storage_dtype": STORAGE_DTYPE,
            "pooling": POOLING,
            "padding_side": PADDING_SIDE,
            "add_special_tokens": ADD_SPECIAL_TOKENS,
        },
        "addressing": {
            "canonical_type": "utf8_surface_text",
            "donor_token_ids_persisted": False,
            "table_partition": "separate bigram and trigram Engram tables",
            "row_order": "Stage 1 frequency rank within each table",
            "runtime_addressing": (
                "stage3_jtd/build_joint_transfer_domain.py compiles each "
                "table's keys.jsonl with the exact resolved Qwen tokenizer; "
                "the donor vectors and row order stay unchanged"
            ),
        },
        "input": current_input,
        "tables": table_artifacts,
        "sharding": {
            "shard_size": shard_size,
            "tensor_keys": [f"layer{layer:02d}" for layer in HIDDEN_STATE_LAYERS],
            "both_layer_views_share_each_engram_row": True,
        },
        "runtime": {
            "gpu": "A100-80GB",
            "requested_batch_size": batch_size,
            "max_padded_tokens": max_padded_tokens,
            "new_shards_this_invocation": shard_stats,
        },
        "universal_subspace_applied": False,
    }
    atomic_write_text(
        completed_manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    corpus_volume.commit()
    print(
        f"[complete] {table_artifacts['bigrams']['rows']:,} bigram Engrams "
        f"and {table_artifacts['trigrams']['rows']:,} trigram Engrams; "
        "every row contains layer08 and layer24"
    )
    return manifest


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    smoke: bool = False,
    force: bool = False,
    shard_size: int = DEFAULT_SHARD_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_padded_tokens: int = DEFAULT_MAX_PADDED_TOKENS,
) -> None:
    mode = "smoke" if smoke else "full"
    print(f"Preparing Stage 2 assets ({mode} run)...")
    prepared = prepare_assets.remote(smoke=smoke)
    result = extract_banks.remote(
        prepared=prepared,
        smoke=smoke,
        force=force,
        shard_size=shard_size,
        batch_size=batch_size,
        max_padded_tokens=max_padded_tokens,
    )
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "bigram_rows": result["tables"]["bigrams"]["rows"],
                "trigram_rows": result["tables"]["trigrams"]["rows"],
                "layers": result["states"]["hidden_state_layers"],
                "output": str(fixed_output_root(smoke)),
                "model_revision": result["model"]["resolved_revision"],
            },
            indent=2,
        )
    )
