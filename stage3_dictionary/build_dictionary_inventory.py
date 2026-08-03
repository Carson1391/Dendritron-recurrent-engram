"""Build Dendritron's word/sense inventory before donor extraction.

The canonical source is the manifest created by
``prepare_definition_sources.py``. Additional reviewed definitions can be
supplied as JSONL records with fields such as:

    {"word": "quark", "pos": "noun", "definition": "...", "source": "..."}

The output keeps one immutable row per dictionary sense and creates ordered
graph edges from every sense to the individual words occurring in its
definition. Definition words always receive stable word IDs. Coverage against
every word in the frozen Engram vocabulary is mandatory.

Examples
--------
Canonical definitions plus the vocabulary already present in both Engram banks:

    python stage3_dictionary/build_dictionary_inventory.py \
        --definition-manifest \
          /data/dendritron-stage3-definition-sources/canonical/definition_sources_manifest.json \
        --ngram-keys /data/dendritron-stage2/bigrams/keys.jsonl \
        --ngram-keys /data/dendritron-stage2/trigrams/keys.jsonl \
        --output /data/dendritron-stage3-dictionary/inventory

Add a reviewed supplemental dictionary export:

    python stage3_dictionary/build_dictionary_inventory.py \
        --definition-manifest definition_sources_manifest.json \
        --definitions science_definitions.jsonl \
        --output dictionary_inventory
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sqlite3
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dendritron.definition_bank import (
    DEFINITION_READOUT_MARKER,
    DefinitionSense,
    DictionaryWord,
    build_dictionary_records,
    iter_definition_words,
    normalize_word,
    read_jsonl,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def iter_jsonl_definitions(paths: Sequence[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for record in read_jsonl(handle):
                record.setdefault("source", path.stem)
                yield record


def load_definition_manifest(path: Path) -> tuple[list[Path], dict[str, Any]]:
    """Resolve and validate the canonical definition artifacts."""
    if not path.is_file():
        raise FileNotFoundError(f"Definition source manifest is unavailable: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 2:
        raise ValueError(f"Unsupported definition source manifest: {path}")
    selection_policy = manifest.get("selection_policy", {})
    if (
        selection_policy.get("payload_scope")
        != "complete_curated_single_word_dictionary"
        or selection_policy.get("definition_word_policy")
        != "ordered_word_id_links"
    ):
        raise ValueError(
            "Definition source manifest lacks the complete dictionary "
            f"selection contract: {path}"
        )
    artifacts = manifest.get("canonical_definition_files", ())
    if not artifacts:
        raise ValueError(f"Definition source manifest has no canonical files: {path}")
    resolved: list[Path] = []
    for artifact in artifacts:
        source_path = Path(artifact["path"])
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Canonical definition artifact is unavailable: {source_path}"
            )
        actual_sha256 = file_sha256(source_path)
        if actual_sha256 != artifact["sha256"]:
            raise RuntimeError(
                f"Canonical definition artifact changed: {source_path}"
            )
        resolved.append(source_path)
    return resolved, manifest


def validate_ngram_contract(
    manifest: Mapping[str, Any],
    ngram_paths: Sequence[Path],
) -> None:
    """Ensure the selected definitions belong to these exact Engram keys."""
    recorded = manifest.get("ngram_key_files", ())
    if len(recorded) != len(ngram_paths):
        raise ValueError(
            "Definition source manifest and inventory use different Engram "
            "key-file counts"
        )
    for expected, actual_path in zip(recorded, ngram_paths, strict=True):
        if not actual_path.is_file():
            raise FileNotFoundError(actual_path)
        if file_sha256(actual_path) != expected.get("sha256"):
            raise RuntimeError(
                "Definition selection belongs to different Stage-2 Engram "
                f"keys: {actual_path}"
            )


def iter_wordnet_definitions(download: bool) -> Iterator[dict[str, Any]]:
    try:
        import nltk
        from nltk.corpus import wordnet
    except ImportError as error:
        raise RuntimeError(
            "WordNet import requires nltk. Install nltk>=3.9."
        ) from error

    if download:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    try:
        synsets = wordnet.all_synsets()
        first = next(synsets)
        synsets = itertools.chain((first,), synsets)
    except LookupError as error:
        raise RuntimeError(
            "NLTK WordNet data is unavailable. Add --download-wordnet once."
        ) from error

    for synset in synsets:
        definition = synset.definition()
        examples = tuple(synset.examples())
        synset_id = synset.name()
        for lemma in synset.lemmas():
            surface = lemma.name().replace("_", " ")
            words = list(iter_definition_words(surface))
            if len(words) != 1 or words[0] != surface:
                continue
            try:
                sense_key = lemma.key()
            except Exception:
                sense_key = f"{synset_id}:{surface}"
            yield {
                "surface": surface,
                "part_of_speech": synset.pos(),
                "definition": definition,
                "source": "princeton_wordnet",
                "source_sense_key": sense_key,
                "synset_id": synset_id,
                "examples": examples,
            }


def iter_ngram_words(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, record in enumerate(read_jsonl(handle), start=1):
                text = record.get("text", record.get("surface_text"))
                if text is None:
                    raise KeyError(
                        f"{path}:{line_number}: expected text or surface_text"
                    )
                yield from iter_definition_words(str(text))


def write_inventory_jsonl(
    output: Path,
    words: Sequence[DictionaryWord],
    senses: Sequence[DefinitionSense],
) -> tuple[Path, Path]:
    words_path = output / "words.jsonl"
    senses_path = output / "senses.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    with words_path.open("w", encoding="utf-8") as handle:
        write_jsonl((word.to_record() for word in words), handle)
    with senses_path.open("w", encoding="utf-8") as handle:
        write_jsonl((sense.to_record() for sense in senses), handle)
    return words_path, senses_path


def build_lookup_database(
    path: Path,
    words: Sequence[DictionaryWord],
    senses: Sequence[DefinitionSense],
) -> dict[str, int]:
    temporary = path.with_name(path.name + ".tmp")
    for candidate in (
        temporary,
        temporary.with_name(temporary.name + "-wal"),
        temporary.with_name(temporary.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)
    connection = sqlite3.connect(str(temporary))
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE words (
                word_id INTEGER PRIMARY KEY,
                surface TEXT NOT NULL,
                normalized TEXT NOT NULL UNIQUE,
                in_ngram_vocabulary INTEGER NOT NULL CHECK (
                    in_ngram_vocabulary IN (0, 1)
                )
            );

            CREATE TABLE senses (
                sense_row INTEGER PRIMARY KEY,
                sense_id TEXT NOT NULL UNIQUE,
                word_id INTEGER NOT NULL REFERENCES words(word_id),
                part_of_speech TEXT NOT NULL,
                definition TEXT NOT NULL,
                source TEXT NOT NULL,
                source_sense_key TEXT NOT NULL,
                examples_json TEXT NOT NULL,
                UNIQUE(source, source_sense_key, word_id)
            );
            CREATE INDEX senses_by_word ON senses(word_id);

            CREATE TABLE definition_word_edges (
                sense_row INTEGER NOT NULL REFERENCES senses(sense_row),
                position INTEGER NOT NULL,
                definition_surface TEXT NOT NULL,
                definition_word_id INTEGER NOT NULL REFERENCES words(word_id),
                PRIMARY KEY(sense_row, position)
            ) WITHOUT ROWID;
            CREATE INDEX definition_edges_by_word
                ON definition_word_edges(definition_word_id);
            """
        )
        word_id_by_normalized = {word.normalized: word.word_id for word in words}
        with connection:
            connection.executemany(
                """
                INSERT INTO words(
                    word_id, surface, normalized, in_ngram_vocabulary
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        word.word_id,
                        word.surface,
                        word.normalized,
                        int(word.in_ngram_vocabulary),
                    )
                    for word in words
                ),
            )
            connection.executemany(
                """
                INSERT INTO senses(
                    sense_row,
                    sense_id,
                    word_id,
                    part_of_speech,
                    definition,
                    source,
                    source_sense_key,
                    examples_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    (
                        sense.sense_row,
                        sense.sense_id,
                        sense.word_id,
                        sense.part_of_speech,
                        sense.definition,
                        sense.source,
                        sense.source_sense_key,
                        json.dumps(
                            list(sense.examples),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                    for sense in senses
                ),
            )
            connection.executemany(
                """
                INSERT INTO definition_word_edges(
                    sense_row,
                    position,
                    definition_surface,
                    definition_word_id
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    (
                        sense.sense_row,
                        position,
                        definition_surface,
                        word_id_by_normalized[normalize_word(definition_surface)],
                    )
                    for sense in senses
                    for position, definition_surface in enumerate(
                        sense.definition_words
                    )
                ),
            )
            connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", "1"),
                    ("definition_vector_layer", "2"),
                    ("definition_readout_marker", DEFINITION_READOUT_MARKER),
                    ("created_at_utc", utc_now()),
                ),
            )
        counts = {
            "words": len(words),
            "senses": len(senses),
            "definition_word_edges": int(
                connection.execute(
                    "SELECT count(*) FROM definition_word_edges"
                ).fetchone()[0]
            ),
        }
    finally:
        connection.close()
    temporary.replace(path)
    return counts


def build_inventory(args: argparse.Namespace) -> dict[str, Any]:
    definition_manifest_path = getattr(args, "definition_manifest", None)
    definition_manifest: dict[str, Any] | None = None
    canonical_paths: list[Path] = []
    if definition_manifest_path is not None:
        canonical_paths, definition_manifest = load_definition_manifest(
            Path(definition_manifest_path)
        )
        validate_ngram_contract(definition_manifest, args.ngram_keys)
    definition_paths = canonical_paths + list(args.definitions)

    candidate_streams: list[Iterable[Mapping[str, Any]]] = []
    if args.wordnet:
        candidate_streams.append(iter_wordnet_definitions(args.download_wordnet))
    if definition_paths:
        candidate_streams.append(iter_jsonl_definitions(definition_paths))
    if not candidate_streams:
        raise ValueError(
            "Choose --definition-manifest, --wordnet, and/or at least one "
            "--definitions file"
        )

    words, senses, report = build_dictionary_records(
        itertools.chain.from_iterable(candidate_streams),
        ngram_words=iter_ngram_words(args.ngram_keys),
    )
    words_path, senses_path = write_inventory_jsonl(args.output, words, senses)
    database_path = args.output / "dictionary.sqlite3"
    database_counts = build_lookup_database(database_path, words, senses)

    report.update(
        {
            "created_at_utc": utc_now(),
            "input_definition_files": [str(path) for path in definition_paths],
            "input_ngram_key_files": [str(path) for path in args.ngram_keys],
            "input_definition_artifacts": [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for path in definition_paths
            ],
            "input_ngram_key_artifacts": [
                {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for path in args.ngram_keys
            ],
            "wordnet_enabled": bool(args.wordnet),
            "definition_source_manifest": (
                {
                    "path": str(definition_manifest_path),
                    "sha256": file_sha256(Path(definition_manifest_path)),
                    "coverage": definition_manifest.get("coverage", {}),
                    "parser_version": definition_manifest.get("parser_version"),
                    "selection_policy": definition_manifest.get(
                        "selection_policy", {}
                    ),
                }
                if definition_manifest is not None
                else None
            ),
            "payload_selection": {
                "scope": "complete_curated_single_word_dictionary",
                "retain_all_word_senses": True,
                "definition_words": "ordered_word_id_links",
            },
            "storage": {
                "definition_vector_width": 2048,
                "definition_vector_dtype": "bfloat16",
                "definition_vector_bank_bytes": len(senses) * 2048 * 2,
                "definition_vector_bank_gib": (
                    len(senses) * 2048 * 2 / (1024**3)
                ),
                "metadata_artifacts_bytes": (
                    words_path.stat().st_size
                    + senses_path.stat().st_size
                    + database_path.stat().st_size
                ),
                "universal_coordinate_bank_bytes": 0,
            },
            "database_counts": database_counts,
            "artifacts": {
                "words_jsonl": {
                    "path": str(words_path),
                    "sha256": file_sha256(words_path),
                },
                "senses_jsonl": {
                    "path": str(senses_path),
                    "sha256": file_sha256(senses_path),
                },
                "dictionary_sqlite3": {
                    "path": str(database_path),
                    "sha256": file_sha256(database_path),
                },
            },
        }
    )
    report_path = args.output / "inventory_manifest.json"
    atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definition-manifest",
        type=Path,
        help=(
            "Manifest produced by prepare_definition_sources.py; its canonical "
            "files are verified and loaded in locked order"
        ),
    )
    parser.add_argument("--wordnet", action="store_true")
    parser.add_argument("--download-wordnet", action="store_true")
    parser.add_argument(
        "--definitions",
        type=Path,
        action="append",
        default=[],
        help="Additional JSONL definition source; repeatable",
    )
    parser.add_argument(
        "--ngram-keys",
        type=Path,
        action="append",
        default=[],
        help="Stage 2 keys.jsonl file; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_inventory(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
