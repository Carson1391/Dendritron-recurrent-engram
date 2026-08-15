"""
Dendritron word-unit front end.

The Qwen tokenizer remains fixed. This module builds:

1. ``token_units``: one row for every Qwen vocabulary ID.
2. ``word_units``: one row for every distinct complete word observed in input.
3. ``word_forms``: every Qwen token-ID sequence observed for that word.

There is no frequency filter. ``encode_input`` inserts every Unicode word from
the current input before returning the token/word alignment used by the model.

Install:
    pip install "transformers>=5.4,<6"

Initialize with the same tokenizer used by the donor:
    python word_unit_table.py init \
        --database word_units.sqlite3 \
        --tokenizer Qwen/Qwen3.6-35B-A3B

Ingest training text:
    python word_unit_table.py ingest \
        --database word_units.sqlite3 \
        --tokenizer Qwen/Qwen3.6-35B-A3B \
        --input training.txt

Inspect one word:
    python word_unit_table.py inspect \
        --database word_units.sqlite3 \
        --word camera
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DEFAULT_TOKENIZER = "Qwen/Qwen3.6-35B-A3B"
SCHEMA_VERSION = 1

INTERNAL_WORD_CONNECTORS = frozenset(("'", "’", "-", "\u2011"))


def _is_word_start(character: str) -> bool:
    return unicodedata.category(character)[:1] in {"L", "N"}


def _is_word_body(character: str) -> bool:
    return unicodedata.category(character)[:1] in {"L", "M", "N"}


def iter_word_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield Unicode word spans with internal apostrophes and hyphens."""
    cursor = 0
    while cursor < len(text):
        if not _is_word_start(text[cursor]):
            cursor += 1
            continue

        start = cursor
        cursor += 1
        while cursor < len(text):
            character = text[cursor]
            if _is_word_body(character):
                cursor += 1
                continue
            if (
                character in INTERNAL_WORD_CONNECTORS
                and cursor + 1 < len(text)
                and _is_word_start(text[cursor + 1])
            ):
                cursor += 2
                while cursor < len(text) and _is_word_body(text[cursor]):
                    cursor += 1
                continue
            break
        yield start, cursor


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _normalize_word(surface: str) -> str:
    return unicodedata.normalize("NFKC", surface).casefold()


def _flatten_single(value: Any, field: str) -> list[Any]:
    """Normalize one-example tokenizer output to a flat Python list."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"Expected one {field} sequence, received {len(value)}")
        value = value[0]
    return list(value)


def _encode_with_offsets(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    token_ids = [int(value) for value in _flatten_single(encoded["input_ids"], "input")]
    offsets = [
        (int(start), int(end))
        for start, end in _flatten_single(encoded["offset_mapping"], "offset")
    ]
    if len(token_ids) != len(offsets):
        raise RuntimeError(
            f"Tokenizer returned {len(token_ids)} IDs and {len(offsets)} offsets"
        )
    return token_ids, offsets


def tokenizer_fingerprint(tokenizer: Any) -> str:
    vocabulary = tokenizer.get_vocab()
    payload = {
        "class": type(tokenizer).__name__,
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "vocabulary": sorted(
            ((str(piece), int(token_id)) for piece, token_id in vocabulary.items()),
            key=lambda item: (item[1], item[0]),
        ),
        "special_ids": sorted(int(value) for value in tokenizer.all_special_ids),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WordOccurrence:
    word_id: int
    surface: str
    char_start: int
    char_end: int
    token_positions: tuple[int, ...]
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class LexicalEncoding:
    text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    words: tuple[WordOccurrence, ...]
    # A tokenizer piece can theoretically overlap more than one word. Keeping a
    # tuple per token preserves that relation exactly.
    token_to_word_ids: tuple[tuple[int, ...], ...]


class WordUnitTable:
    """SQLite-backed Qwen-token-to-complete-word address table."""

    def __init__(self, database: str | Path, tokenizer: Any) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.connection = sqlite3.connect(str(self.database))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()
        self._bind_tokenizer()
        self._populate_token_units()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "WordUnitTable":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE IF NOT EXISTS token_units (
                token_id INTEGER PRIMARY KEY,
                token_piece TEXT NOT NULL,
                decoded_text TEXT NOT NULL,
                decoded_bytes BLOB NOT NULL,
                is_special INTEGER NOT NULL CHECK (is_special IN (0, 1))
            );

            CREATE TABLE IF NOT EXISTS word_units (
                word_id INTEGER PRIMARY KEY AUTOINCREMENT,
                surface TEXT NOT NULL UNIQUE,
                normalized TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS word_units_by_normalized
                ON word_units(normalized);

            CREATE TABLE IF NOT EXISTS word_forms (
                form_id INTEGER PRIMARY KEY AUTOINCREMENT,
                word_id INTEGER NOT NULL REFERENCES word_units(word_id),
                context_kind TEXT NOT NULL,
                token_ids_json TEXT NOT NULL,
                relative_offsets_json TEXT NOT NULL,
                UNIQUE(
                    word_id,
                    context_kind,
                    token_ids_json,
                    relative_offsets_json
                )
            );
            CREATE INDEX IF NOT EXISTS word_forms_by_word
                ON word_forms(word_id);
            """
        )
        self.connection.commit()

    def _bind_tokenizer(self) -> None:
        fingerprint = tokenizer_fingerprint(self.tokenizer)
        existing = self.connection.execute(
            "SELECT value FROM metadata WHERE key='tokenizer_fingerprint'"
        ).fetchone()
        if existing is not None and existing[0] != fingerprint:
            raise RuntimeError(
                "This database is already bound to a different tokenizer snapshot"
            )

        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "tokenizer_fingerprint": fingerprint,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name_or_path": str(
                getattr(self.tokenizer, "name_or_path", "")
            ),
            "vocabulary_size": str(len(self.tokenizer.get_vocab())),
            "word_policy": "append_every_observed_unicode_word",
        }
        self.connection.executemany(
            """
            INSERT INTO metadata(key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            metadata.items(),
        )
        self.connection.commit()

    def _populate_token_units(self) -> None:
        expected = len(set(int(value) for value in self.tokenizer.get_vocab().values()))
        present = int(
            self.connection.execute("SELECT count(*) FROM token_units").fetchone()[0]
        )
        if present == expected:
            return
        if present:
            raise RuntimeError(
                f"Partial token table found: expected {expected}, found {present}"
            )

        special_ids = {int(value) for value in self.tokenizer.all_special_ids}
        vocabulary = self.tokenizer.get_vocab()
        pieces_by_id: dict[int, list[str]] = {}
        for piece, token_id in vocabulary.items():
            pieces_by_id.setdefault(int(token_id), []).append(str(piece))

        rows: list[tuple[int, str, str, bytes, int]] = []
        for token_id in sorted(pieces_by_id):
            converted = self.tokenizer.convert_ids_to_tokens(token_id)
            piece = (
                str(converted)
                if converted is not None
                else sorted(pieces_by_id[token_id])[0]
            )
            decoded = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            decoded = str(decoded)
            rows.append(
                (
                    token_id,
                    piece,
                    decoded,
                    decoded.encode("utf-8", errors="surrogatepass"),
                    int(token_id in special_ids),
                )
            )

        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO token_units(
                    token_id,
                    token_piece,
                    decoded_text,
                    decoded_bytes,
                    is_special
                ) VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def _get_or_create_word(self, surface: str) -> int:
        row = self.connection.execute(
            "SELECT word_id FROM word_units WHERE surface=?",
            (surface,),
        ).fetchone()
        if row is not None:
            return int(row[0])
        cursor = self.connection.execute(
            "INSERT INTO word_units(surface, normalized) VALUES (?, ?)",
            (surface, _normalize_word(surface)),
        )
        return int(cursor.lastrowid)

    def _store_form(
        self,
        word_id: int,
        context_kind: str,
        token_ids: Sequence[int],
        relative_offsets: Sequence[tuple[int, int]],
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO word_forms(
                word_id,
                context_kind,
                token_ids_json,
                relative_offsets_json
            ) VALUES (?, ?, ?, ?)
            """,
            (
                word_id,
                context_kind,
                _json([int(value) for value in token_ids]),
                _json([[int(start), int(end)] for start, end in relative_offsets]),
            ),
        )

    def _store_canonical_forms(self, word_id: int, surface: str) -> None:
        for context_kind, text, shift in (
            ("standalone", surface, 0),
            ("after_space", " " + surface, 1),
        ):
            token_ids, offsets = _encode_with_offsets(self.tokenizer, text)
            self._store_form(
                word_id,
                context_kind,
                token_ids,
                [(start - shift, end - shift) for start, end in offsets],
            )

    def encode_input(self, text: str) -> LexicalEncoding:
        """
        Tokenize one model input and persist every complete word it contains.

        The returned alignment is computed from the actual full-input
        tokenization, so it remains correct for BOS, whitespace, punctuation,
        Unicode, and alternate tokenization forms.
        """
        token_ids, offsets = _encode_with_offsets(self.tokenizer, text)
        token_to_words: list[list[int]] = [[] for _ in token_ids]
        occurrences: list[WordOccurrence] = []

        with self.connection:
            for char_start, char_end in iter_word_spans(text):
                surface = text[char_start:char_end]
                word_id = self._get_or_create_word(surface)
                self._store_canonical_forms(word_id, surface)

                positions = [
                    index
                    for index, (token_start, token_end) in enumerate(offsets)
                    if token_end > char_start and token_start < char_end
                ]
                if not positions:
                    raise RuntimeError(
                        f"No tokenizer piece overlaps word {surface!r} at "
                        f"{char_start}:{char_end}"
                    )

                observed_ids = [token_ids[index] for index in positions]
                relative_offsets = [
                    (
                        offsets[index][0] - char_start,
                        offsets[index][1] - char_start,
                    )
                    for index in positions
                ]
                self._store_form(
                    word_id,
                    "observed",
                    observed_ids,
                    relative_offsets,
                )
                for index in positions:
                    token_to_words[index].append(word_id)

                occurrences.append(
                    WordOccurrence(
                        word_id=word_id,
                        surface=surface,
                        char_start=char_start,
                        char_end=char_end,
                        token_positions=tuple(positions),
                        token_ids=tuple(observed_ids),
                    )
                )

        return LexicalEncoding(
            text=text,
            input_ids=tuple(token_ids),
            offsets=tuple(offsets),
            words=tuple(occurrences),
            token_to_word_ids=tuple(tuple(values) for values in token_to_words),
        )

    def inspect_word(self, surface: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT word_id, surface, normalized
            FROM word_units
            WHERE surface=?
            """,
            (surface,),
        ).fetchone()
        if row is None:
            return None
        forms = self.connection.execute(
            """
            SELECT context_kind, token_ids_json, relative_offsets_json
            FROM word_forms
            WHERE word_id=?
            ORDER BY context_kind, form_id
            """,
            (int(row[0]),),
        ).fetchall()
        return {
            "word_id": int(row[0]),
            "surface": row[1],
            "normalized": row[2],
            "forms": [
                {
                    "context": context_kind,
                    "token_ids": json.loads(token_ids_json),
                    "relative_offsets": json.loads(relative_offsets_json),
                }
                for context_kind, token_ids_json, relative_offsets_json in forms
            ],
        }

    def counts(self) -> dict[str, int]:
        return {
            "tokens": int(
                self.connection.execute("SELECT count(*) FROM token_units").fetchone()[0]
            ),
            "words": int(
                self.connection.execute("SELECT count(*) FROM word_units").fetchone()[0]
            ),
            "word_forms": int(
                self.connection.execute("SELECT count(*) FROM word_forms").fetchone()[0]
            ),
        }


def load_tokenizer(identifier: str, revision: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        identifier,
        revision=revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "The tokenizer must expose offset_mapping for exact token-to-word alignment"
        )
    return tokenizer


def _iter_text_inputs(
    paths: Sequence[Path],
    literal_texts: Sequence[str],
    jsonl_field: str | None,
) -> Iterator[str]:
    yield from literal_texts
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            if jsonl_field:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    record = json.loads(raw_line)
                    if jsonl_field not in record:
                        raise KeyError(
                            f"{path}:{line_number}: missing field {jsonl_field!r}"
                        )
                    yield str(record[jsonl_field])
            else:
                # One line is one input episode. This keeps memory bounded and
                # preserves the tokenizer's actual line-level context.
                for raw_line in handle:
                    text = raw_line.rstrip("\r\n")
                    if text:
                        yield text


def _open_table(args: argparse.Namespace) -> WordUnitTable:
    tokenizer = load_tokenizer(args.tokenizer, args.revision)
    return WordUnitTable(args.database, tokenizer)


def _command_init(args: argparse.Namespace) -> None:
    with _open_table(args) as table:
        print(json.dumps(table.counts(), indent=2))


def _command_ingest(args: argparse.Namespace) -> None:
    with _open_table(args) as table:
        inputs = 0
        word_occurrences = 0
        for text in _iter_text_inputs(
            [Path(value) for value in args.input],
            args.text,
            args.jsonl_field,
        ):
            encoding = table.encode_input(text)
            inputs += 1
            word_occurrences += len(encoding.words)
        result = {
            "inputs_ingested": inputs,
            "word_occurrences_ingested": word_occurrences,
            **table.counts(),
        }
        print(json.dumps(result, indent=2))


def _command_inspect(args: argparse.Namespace) -> None:
    connection = sqlite3.connect(str(args.database))
    try:
        row = connection.execute(
            """
            SELECT word_id, surface, normalized
            FROM word_units
            WHERE surface=?
            """,
            (args.word,),
        ).fetchone()
        if row is None:
            print(json.dumps({"word": args.word, "present": False}, indent=2))
            return
        forms = connection.execute(
            """
            SELECT context_kind, token_ids_json, relative_offsets_json
            FROM word_forms
            WHERE word_id=?
            ORDER BY context_kind, form_id
            """,
            (int(row[0]),),
        ).fetchall()
        print(
            json.dumps(
                {
                    "present": True,
                    "word_id": int(row[0]),
                    "surface": row[1],
                    "normalized": row[2],
                    "forms": [
                        {
                            "context": item[0],
                            "token_ids": json.loads(item[1]),
                            "relative_offsets": json.loads(item[2]),
                        }
                        for item in forms
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        connection.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_tokenizer_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--database", type=Path, required=True)
        command.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
        command.add_argument("--revision")

    initialize = subparsers.add_parser("init", help="Build the full token-unit table")
    add_tokenizer_arguments(initialize)
    initialize.set_defaults(function=_command_init)

    ingest = subparsers.add_parser(
        "ingest",
        help="Append every word found in the supplied model inputs",
    )
    add_tokenizer_arguments(ingest)
    ingest.add_argument("--input", action="append", default=[])
    ingest.add_argument("--text", action="append", default=[])
    ingest.add_argument("--jsonl-field")
    ingest.set_defaults(function=_command_ingest)

    inspect = subparsers.add_parser("inspect", help="Show one stored word and its forms")
    inspect.add_argument("--database", type=Path, required=True)
    inspect.add_argument("--word", required=True)
    inspect.set_defaults(function=_command_inspect)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest" and not args.input and not args.text:
        parser.error("ingest requires at least one --input or --text")
    args.function(args)


if __name__ == "__main__":
    main()
