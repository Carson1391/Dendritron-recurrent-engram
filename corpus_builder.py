"""
Stage 1: build the frozen text-key inventory for the engram tables.

This job does NOT load Qwen, tokenize with a model tokenizer, assign token IDs,
or extract hidden states. It:

1. Builds a science-first 200M-word corpus:
   - 140M words from selected arXiv title + abstract records
   - 40M words from unfiltered WikiText-103
   - 20M words from FineMath-4+
2. Cleans formatting noise and exactly deduplicates documents.
3. Counts every word bigram and trigram with raw frequency, resetting the
   counting window at punctuation and symbol boundaries.
4. Flushes complete partial counters to Parquet without pruning.
5. Uses DuckDB to sum every partial count and select the global top 500,000
   bigrams and top 500,000 trigrams.
6. Saves UTF-8 text keys, not tokenizer IDs.

All three datasets are streamed from Hugging Face inside the Modal function.
The arXiv stream is filtered to the selected science categories before its
title and abstract text enters the n-gram counters.

Run:
    pip install modal
    modal setup
    modal run corpus_builder.py

Resume:
    Run the same command again. Completed checkpoints are reused.

Clean rebuild:
    modal run corpus_builder.py --force

Outputs on the "dendritron-corpus" Modal Volume:
    /dendritron-stage1/final/top_bigrams.jsonl
    /dendritron-stage1/final/top_trigrams.jsonl
    /dendritron-stage1/final/corpus_statistics.json
    /dendritron-stage1/final/preview_top_100_bigrams.txt
    /dendritron-stage1/final/preview_top_100_trigrams.txt
"""

from __future__ import annotations

import hashlib
import html
import itertools
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import modal

from dendritron.tokenizer import complete_word_segments


# ---------------------------------------------------------------------------
# Locked corpus specification
# ---------------------------------------------------------------------------

APP_NAME = "dendritron-corpus-builder"
VOLUME_NAME = "dendritron-corpus"
VOLUME_ROOT = Path("/data/dendritron-stage1")
WORK_DIR = VOLUME_ROOT / "work"
PARTIAL_DIR = WORK_DIR / "partials"
HASH_DIR = WORK_DIR / "document-hashes"
STATE_DIR = WORK_DIR / "state"
FINAL_DIR = VOLUME_ROOT / "final"

SOURCE_TOKEN_BUDGETS = {
    "arxiv_science": 140_000_000,
    "wikitext_103": 40_000_000,
    "finemath_4plus": 20_000_000,
}
TOTAL_TOKEN_BUDGET = sum(SOURCE_TOKEN_BUDGETS.values())

TOP_K = 500_000
BIGRAM_FLUSH_THRESHOLD = 10_000_000
TRIGRAM_FLUSH_THRESHOLD = 5_000_000
CHECKPOINT_EVERY_ROWS = 25_000
PARQUET_WRITE_BATCH_ROWS = 250_000
SHUFFLE_BUFFER_SIZE = 10_000
SHUFFLE_SEED = 42

ARXIV_DATASET = "librarian-bots/arxiv-metadata-snapshot"
ARXIV_CONFIG = "default"

ARXIV_EXACT_CATEGORIES = {
    "gr-qc",
    "quant-ph",
    "hep-th",
    "hep-ph",
    "hep-ex",
    "hep-lat",
    "nucl-th",
    "nucl-ex",
    "math-ph",
    "nlin.CD",
}
ARXIV_CATEGORY_PREFIXES = {
    "physics.",
    "astro-ph.",
    "cond-mat.",
    "q-bio.",
}

WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
FINEMATH_DATASET = "HuggingFaceTB/finemath"
FINEMATH_CONFIG = "finemath-4plus"

# Internal separator only. It cannot occur in a token produced by WORD_RE.
NGRAM_SEPARATOR = "\x1f"


# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

app = modal.App(APP_NAME)
corpus_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "datasets>=4.0,<5",
        "duckdb>=1.4,<2",
        "pyarrow>=18,<24",
        "regex>=2024.11.6",
    )
    .env(
        {
            "HF_HOME": "/tmp/huggingface",
            "HF_DATASETS_CACHE": "/tmp/huggingface/datasets",
        }
    )
)


# ---------------------------------------------------------------------------
# Text cleanup and word tokenization
# ---------------------------------------------------------------------------

# Unicode words and numbers, including internal apostrophes/hyphens. Punctuation
# is deliberately excluded because these are word n-grams, not tokenizer n-grams.
WORD_PATTERN = (
    r"(?:\p{L}|\p{N})[\p{L}\p{M}\p{N}]*"
    r"(?:['’\-\u2011](?:\p{L}|\p{N})[\p{L}\p{M}\p{N}]*)*"
)

URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WHITESPACE_RE = re.compile(r"\s+")
WIKITEXT_JOIN_REPLACEMENTS = {
    " @-@ ": "-",
    " @,@ ": ",",
    " @.@ ": ".",
}

# Remove markup commands while retaining the words inside their arguments.
# Semantic commands not listed here (for example \alpha) become the word "alpha".
LATEX_FORMAT_COMMAND_RE = re.compile(
    r"\\(?:"
    r"begin|end|left|right|text|textrm|textbf|textit|mathrm|mathbf|mathit|"
    r"mathbb|mathcal|operatorname|emph|cite|citep|citet|ref|eqref|label|"
    r"section|subsection|subsubsection|paragraph|frac|dfrac|tfrac|sqrt|"
    r"displaystyle|limits|nonumber|notag|quad|qquad|hspace|vspace"
    r")\*?(?:\[[^\]]*\])?",
    re.IGNORECASE,
)
LATEX_COMMAND_RE = re.compile(r"\\([A-Za-z]+)")


def clean_text(value: Any) -> str:
    """Conservative formatting cleanup; no semantic/domain filtering."""
    if value is None:
        return ""
    text = html.unescape(str(value))
    for old, new in WIKITEXT_JOIN_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = URL_RE.sub(" ", text)
    text = LATEX_FORMAT_COMMAND_RE.sub(" ", text)
    text = LATEX_COMMAND_RE.sub(r" \1 ", text)
    text = CONTROL_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def compile_word_regex():
    # Imported inside the Modal image so the local launcher only needs `modal`.
    import regex

    return regex.compile(WORD_PATTERN, regex.VERSION1)


def document_digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


# ---------------------------------------------------------------------------
# Durable state
# ---------------------------------------------------------------------------


@dataclass
class SourceState:
    source: str
    spec_fingerprint: str
    rows_seen: int = 0
    documents_used: int = 0
    documents_skipped_duplicate: int = 0
    documents_skipped_empty: int = 0
    documents_skipped_filter: int = 0
    tokens: int = 0
    bigrams: int = 0
    trigrams: int = 0
    shard_sequence: int = 0
    completed: bool = False


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def state_path(source: str) -> Path:
    return STATE_DIR / f"{source}.json"


def load_source_state(source: str, fingerprint: str) -> SourceState:
    path = state_path(source)
    if not path.exists():
        return SourceState(source=source, spec_fingerprint=fingerprint)

    payload = json.loads(path.read_text(encoding="utf-8"))
    state = SourceState(**payload)
    if state.spec_fingerprint != fingerprint:
        raise RuntimeError(
            f"The saved state for {source!r} was made with a different corpus "
            "specification. Run with --force to rebuild it."
        )
    return state


def save_source_state(state: SourceState) -> None:
    atomic_write_text(
        state_path(state.source),
        json.dumps(asdict(state), indent=2, sort_keys=True) + "\n",
    )


def parse_sequence(path: Path) -> int | None:
    match = re.search(r"_(\d{6})\.(?:parquet|bin)$", path.name)
    return int(match.group(1)) if match else None


def remove_uncommitted_shards(source: str, committed_sequence: int) -> None:
    patterns = (
        (PARTIAL_DIR, f"bigram_{source}_*.parquet"),
        (PARTIAL_DIR, f"trigram_{source}_*.parquet"),
        (HASH_DIR, f"hashes_{source}_*.bin"),
    )
    for directory, pattern in patterns:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            sequence = parse_sequence(path)
            if sequence is not None and sequence > committed_sequence:
                path.unlink()


def load_seen_document_hashes() -> set[bytes]:
    seen: set[bytes] = set()
    if not HASH_DIR.exists():
        return seen

    for path in sorted(HASH_DIR.glob("hashes_*.bin")):
        data = path.read_bytes()
        if len(data) % 16:
            raise RuntimeError(f"Corrupt document-hash shard: {path}")
        seen.update(data[index : index + 16] for index in range(0, len(data), 16))
    return seen


def specification_fingerprint() -> str:
    locked_spec = {
        "source_token_budgets": SOURCE_TOKEN_BUDGETS,
        "top_k": TOP_K,
        "arxiv_exact_categories": sorted(ARXIV_EXACT_CATEGORIES),
        "arxiv_category_prefixes": sorted(ARXIV_CATEGORY_PREFIXES),
        "arxiv": [ARXIV_DATASET, ARXIV_CONFIG],
        "wikitext": [WIKITEXT_DATASET, WIKITEXT_CONFIG],
        "finemath": [FINEMATH_DATASET, FINEMATH_CONFIG],
        "word_pattern": WORD_PATTERN,
        "shuffle_seed": SHUFFLE_SEED,
        "counting": "exact_raw_frequency_no_partial_pruning_punctuation_boundaries",
    }
    encoded = json.dumps(locked_spec, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Exact partial-counter writer
# ---------------------------------------------------------------------------


def write_counter_parquet(counter: Counter[str], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    schema = pa.schema(
        [
            pa.field("text_key", pa.string(), nullable=False),
            pa.field("frequency", pa.uint64(), nullable=False),
        ]
    )

    with pq.ParquetWriter(
        temporary,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    ) as writer:
        iterator = iter(counter.items())
        while True:
            batch = list(itertools.islice(iterator, PARQUET_WRITE_BATCH_ROWS))
            if not batch:
                break
            table = pa.Table.from_arrays(
                [
                    pa.array((item[0] for item in batch), type=pa.string()),
                    pa.array((item[1] for item in batch), type=pa.uint64()),
                ],
                schema=schema,
            )
            writer.write_table(table, row_group_size=PARQUET_WRITE_BATCH_ROWS)

    os.replace(temporary, path)


class ExactNgramAccumulator:
    def __init__(self) -> None:
        self.bigrams: Counter[str] = Counter()
        self.trigrams: Counter[str] = Counter()

    def add(self, words: list[str]) -> tuple[int, int]:
        bigram_count = max(0, len(words) - 1)
        trigram_count = max(0, len(words) - 2)

        self.bigrams.update(
            left + NGRAM_SEPARATOR + right
            for left, right in zip(words, words[1:])
        )
        self.trigrams.update(
            first + NGRAM_SEPARATOR + second + NGRAM_SEPARATOR + third
            for first, second, third in zip(words, words[1:], words[2:])
        )
        return bigram_count, trigram_count

    def add_segments(self, segments: Iterable[Iterable[str]]) -> tuple[int, int]:
        bigrams = 0
        trigrams = 0
        for segment in segments:
            current_bigrams, current_trigrams = self.add(list(segment))
            bigrams += current_bigrams
            trigrams += current_trigrams
        return bigrams, trigrams

    def needs_flush(self) -> bool:
        return (
            len(self.bigrams) >= BIGRAM_FLUSH_THRESHOLD
            or len(self.trigrams) >= TRIGRAM_FLUSH_THRESHOLD
        )

    def flush(self, source: str, sequence: int) -> None:
        if self.bigrams:
            write_counter_parquet(
                self.bigrams,
                PARTIAL_DIR / f"bigram_{source}_{sequence:06d}.parquet",
            )
        if self.trigrams:
            write_counter_parquet(
                self.trigrams,
                PARTIAL_DIR / f"trigram_{source}_{sequence:06d}.parquet",
            )
        self.bigrams.clear()
        self.trigrams.clear()


# ---------------------------------------------------------------------------
# Source filters and iterators
# ---------------------------------------------------------------------------


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_streaming_dataset(
    dataset_name: str,
    config_name: str,
    skip_rows: int,
):
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_name,
        config_name,
        split="train",
        streaming=True,
    )
    dataset = dataset.shuffle(
        seed=SHUFFLE_SEED,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )
    if skip_rows:
        # Shuffle first, then skip, so the resumed stream has the same order.
        dataset = dataset.skip(skip_rows)
    return dataset


def iter_wikitext_rows(skip_rows: int) -> Iterable[Mapping[str, Any]]:
    return load_streaming_dataset(
        WIKITEXT_DATASET,
        WIKITEXT_CONFIG,
        skip_rows,
    )


def iter_arxiv_science_rows(skip_rows: int) -> Iterable[Mapping[str, Any]]:
    return load_streaming_dataset(
        ARXIV_DATASET,
        ARXIV_CONFIG,
        skip_rows,
    )


def iter_finemath_rows(skip_rows: int) -> Iterable[Mapping[str, Any]]:
    return load_streaming_dataset(
        FINEMATH_DATASET,
        FINEMATH_CONFIG,
        skip_rows,
    )


def arxiv_category_selected(value: Any) -> bool:
    categories = str(value or "").split()
    for category in categories:
        if category in ARXIV_EXACT_CATEGORIES:
            return True
        for prefix in ARXIV_CATEGORY_PREFIXES:
            if category == prefix[:-1] or category.startswith(prefix):
                return True
    return False


def extract_arxiv_text(row: Mapping[str, Any]) -> str | None:
    if not arxiv_category_selected(row.get("categories")):
        return None
    title = str(row.get("title") or "").strip()
    abstract = str(row.get("abstract") or "").strip()
    text = f"{title}. {abstract}".strip()
    return text or None


def extract_text_field(row: Mapping[str, Any]) -> str | None:
    value = row.get("text")
    return str(value) if value is not None else None


# ---------------------------------------------------------------------------
# Source processing
# ---------------------------------------------------------------------------


def persist_checkpoint(
    state: SourceState,
    accumulator: ExactNgramAccumulator,
    new_hashes: list[bytes],
) -> None:
    next_sequence = state.shard_sequence + 1
    accumulator.flush(state.source, next_sequence)
    if new_hashes:
        atomic_write_bytes(
            HASH_DIR / f"hashes_{state.source}_{next_sequence:06d}.bin",
            b"".join(new_hashes),
        )
        new_hashes.clear()
    state.shard_sequence = next_sequence
    save_source_state(state)
    corpus_volume.commit()


def process_source(
    source: str,
    row_iterator_factory: Callable[[int], Iterable[Mapping[str, Any]]],
    text_extractor: Callable[[Mapping[str, Any]], str | None],
    word_regex,
    seen_hashes: set[bytes],
    fingerprint: str,
) -> SourceState:
    budget = SOURCE_TOKEN_BUDGETS[source]
    state = load_source_state(source, fingerprint)
    remove_uncommitted_shards(source, state.shard_sequence)

    if state.completed:
        print(
            f"{source}: already complete "
            f"({state.tokens:,}/{budget:,} words); reusing checkpoints."
        )
        return state
    if state.tokens >= budget:
        # Handles a preemption after the final checkpoint was committed but
        # before the separate "completed" state update was committed.
        state.completed = True
        save_source_state(state)
        corpus_volume.commit()
        return state

    print(
        f"{source}: resuming at input row {state.rows_seen:,}, "
        f"{state.tokens:,}/{budget:,} words."
    )
    accumulator = ExactNgramAccumulator()
    new_hashes: list[bytes] = []
    rows_since_checkpoint = 0

    for row in row_iterator_factory(state.rows_seen):
        state.rows_seen += 1
        rows_since_checkpoint += 1

        raw_text = text_extractor(row)
        if not raw_text:
            state.documents_skipped_filter += 1
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            state.documents_skipped_empty += 1
            continue

        digest = document_digest(cleaned)
        if digest in seen_hashes:
            state.documents_skipped_duplicate += 1
            continue

        seen_hashes.add(digest)
        new_hashes.append(digest)
        segments = complete_word_segments(cleaned)
        if not segments:
            state.documents_skipped_empty += 1
            continue

        remaining = budget - state.tokens
        selected_segments: list[tuple[str, ...]] = []
        selected_word_count = 0
        for segment in segments:
            available = remaining - selected_word_count
            if available <= 0:
                break
            selected = tuple(segment[:available])
            if selected:
                selected_segments.append(selected)
                selected_word_count += len(selected)

        bigrams, trigrams = accumulator.add_segments(selected_segments)
        state.documents_used += 1
        state.tokens += selected_word_count
        state.bigrams += bigrams
        state.trigrams += trigrams

        should_checkpoint = (
            accumulator.needs_flush()
            or rows_since_checkpoint >= CHECKPOINT_EVERY_ROWS
            or state.tokens >= budget
        )
        if should_checkpoint:
            persist_checkpoint(state, accumulator, new_hashes)
            rows_since_checkpoint = 0
            print(
                f"{source}: {state.tokens:,}/{budget:,} words, "
                f"{state.documents_used:,} documents, "
                f"checkpoint {state.shard_sequence}."
            )

        if state.tokens >= budget:
            state.completed = True
            save_source_state(state)
            corpus_volume.commit()
            break
    else:
        if accumulator.bigrams or accumulator.trigrams or new_hashes:
            persist_checkpoint(state, accumulator, new_hashes)
        raise RuntimeError(
            f"{source} was exhausted at {state.tokens:,} words, below its "
            f"{budget:,}-word budget. The target corpus specification was not met."
        )

    return state


# ---------------------------------------------------------------------------
# Exact global aggregation and exports
# ---------------------------------------------------------------------------


def parquet_file_list_sql(paths: list[Path]) -> str:
    return "[" + ", ".join(sql_string(str(path)) for path in paths) + "]"


def build_top_table(
    connection,
    order_name: str,
    n: int,
    paths: list[Path],
) -> None:
    if not paths:
        raise RuntimeError(f"No {order_name} partial-count files were found.")
    file_list = parquet_file_list_sql(paths)
    connection.execute(
        f"""
        CREATE OR REPLACE TABLE top_{order_name} AS
        SELECT
            replace(text_key, chr(31), ' ') AS text,
            CAST(sum(frequency) AS UBIGINT) AS frequency,
            CAST({n} AS UTINYINT) AS n
        FROM read_parquet({file_list})
        GROUP BY text_key
        ORDER BY frequency DESC, text_key ASC
        LIMIT {TOP_K}
        """
    )


def export_top_table(
    connection,
    order_name: str,
) -> tuple[int, list[str]]:
    output_path = FINAL_DIR / f"top_{order_name}.jsonl"
    preview_path = FINAL_DIR / f"preview_top_100_{order_name}.txt"
    output_tmp = output_path.with_name(output_path.name + ".tmp")
    preview_lines: list[str] = []
    row_count = 0

    cursor = connection.execute(
        f"""
        SELECT text, frequency, n
        FROM top_{order_name}
        ORDER BY frequency DESC, text ASC
        """
    )
    with output_tmp.open("w", encoding="utf-8") as output:
        while True:
            rows = cursor.fetchmany(25_000)
            if not rows:
                break
            for text, frequency, n in rows:
                row_count += 1
                record = {
                    "text": text,
                    "frequency": int(frequency),
                    "n": int(n),
                    "rank": row_count,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                if row_count <= 100:
                    preview_lines.append(
                        f"{row_count}\t{int(frequency)}\t{text}"
                    )

    os.replace(output_tmp, output_path)
    atomic_write_text(preview_path, "\n".join(preview_lines) + "\n")
    return row_count, preview_lines


def aggregate_and_export(
    states: Mapping[str, SourceState],
    fingerprint: str,
) -> dict[str, Any]:
    import duckdb

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    bigram_paths = sorted(PARTIAL_DIR.glob("bigram_*.parquet"))
    trigram_paths = sorted(PARTIAL_DIR.glob("trigram_*.parquet"))

    duckdb_temp = Path("/tmp/duckdb-global")
    duckdb_temp.mkdir(parents=True, exist_ok=True)
    database_path = Path("/tmp/dendritron-stage1.duckdb")
    database_path.unlink(missing_ok=True)

    connection = duckdb.connect(str(database_path))
    connection.execute("SET threads = 8")
    connection.execute("SET memory_limit = '24GB'")
    connection.execute(f"SET temp_directory = {sql_string(str(duckdb_temp))}")
    connection.execute("SET preserve_insertion_order = false")

    print(
        f"Aggregating {len(bigram_paths)} bigram and "
        f"{len(trigram_paths)} trigram partial files exactly..."
    )
    build_top_table(connection, "bigrams", 2, bigram_paths)
    build_top_table(connection, "trigrams", 3, trigram_paths)
    bigram_rows, _ = export_top_table(connection, "bigrams")
    trigram_rows, _ = export_top_table(connection, "trigrams")
    connection.close()

    statistics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec_fingerprint": fingerprint,
        "counting_method": "exact raw frequency; all partial counts globally summed",
        "key_format": "cleaned UTF-8 word n-gram text; no tokenizer IDs",
        "token_definition": "Unicode word/number regex; punctuation-only tokens excluded",
        "total_token_budget": TOTAL_TOKEN_BUDGET,
        "total_tokens_processed": sum(state.tokens for state in states.values()),
        "top_k_per_order": TOP_K,
        "top_bigrams_written": bigram_rows,
        "top_trigrams_written": trigram_rows,
        "partial_bigram_files": len(bigram_paths),
        "partial_trigram_files": len(trigram_paths),
        "source_token_budgets": SOURCE_TOKEN_BUDGETS,
        "sources": {
            source: asdict(state)
            for source, state in states.items()
        },
        "arxiv_exact_categories": sorted(ARXIV_EXACT_CATEGORIES),
        "arxiv_category_prefixes": sorted(ARXIV_CATEGORY_PREFIXES),
        "datasets": {
            "arxiv_science": {
                "source": ARXIV_DATASET,
                "config": ARXIV_CONFIG,
                "split": "train",
                "streaming": True,
                "fields": ["title", "abstract"],
            },
            "wikitext_103": {
                "source": WIKITEXT_DATASET,
                "config": WIKITEXT_CONFIG,
                "split": "train",
                "streaming": True,
            },
            "finemath_4plus": {
                "source": FINEMATH_DATASET,
                "config": FINEMATH_CONFIG,
                "split": "train",
                "streaming": True,
            },
        },
    }
    atomic_write_text(
        FINAL_DIR / "corpus_statistics.json",
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    corpus_volume.commit()
    return statistics


def clean_rebuild_directories() -> None:
    # Fixed, narrow targets under this job's own Volume directory.
    for path in (WORK_DIR, FINAL_DIR):
        if path.exists():
            shutil.rmtree(path)
    corpus_volume.commit()


@app.function(
    image=image,
    cpu=8,
    memory=32_768,
    ephemeral_disk=100 * 1024,
    timeout=24 * 60 * 60,
    volumes={"/data": corpus_volume},
)
def build_corpus(force: bool = False) -> dict[str, Any]:
    if force:
        print("Force rebuild requested: clearing this job's prior work and cache.")
        clean_rebuild_directories()

    for directory in (PARTIAL_DIR, HASH_DIR, STATE_DIR, FINAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    fingerprint = specification_fingerprint()
    word_regex = compile_word_regex()

    # Remove any shard written after the last durable state update, then rebuild
    # the exact dedupe set from committed 16-byte document hashes.
    for source in SOURCE_TOKEN_BUDGETS:
        state = load_source_state(source, fingerprint)
        remove_uncommitted_shards(source, state.shard_sequence)
    seen_hashes = load_seen_document_hashes()
    print(f"Loaded {len(seen_hashes):,} committed document hashes.")

    states: dict[str, SourceState] = {}
    states["arxiv_science"] = process_source(
        source="arxiv_science",
        row_iterator_factory=iter_arxiv_science_rows,
        text_extractor=extract_arxiv_text,
        word_regex=word_regex,
        seen_hashes=seen_hashes,
        fingerprint=fingerprint,
    )
    states["wikitext_103"] = process_source(
        source="wikitext_103",
        row_iterator_factory=iter_wikitext_rows,
        text_extractor=extract_text_field,
        word_regex=word_regex,
        seen_hashes=seen_hashes,
        fingerprint=fingerprint,
    )
    states["finemath_4plus"] = process_source(
        source="finemath_4plus",
        row_iterator_factory=iter_finemath_rows,
        text_extractor=extract_text_field,
        word_regex=word_regex,
        seen_hashes=seen_hashes,
        fingerprint=fingerprint,
    )

    if not all(state.completed for state in states.values()):
        raise RuntimeError("One or more corpus sources did not complete.")
    if sum(state.tokens for state in states.values()) != TOTAL_TOKEN_BUDGET:
        raise RuntimeError("The final corpus did not meet the locked 200M-word budget.")

    statistics = aggregate_and_export(states, fingerprint)
    result = {
        "status": "complete",
        "volume": VOLUME_NAME,
        "output_directory": str(FINAL_DIR),
        "total_tokens": statistics["total_tokens_processed"],
        "top_bigrams": statistics["top_bigrams_written"],
        "top_trigrams": statistics["top_trigrams_written"],
    }
    print(json.dumps(result, indent=2))
    return result


@app.local_entrypoint()
def main(force: bool = False) -> None:
    result = build_corpus.remote(force=force)
    print(json.dumps(result, indent=2))
