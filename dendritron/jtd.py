"""Collision-safe surface lookup used before latent joint transfer.

Stage-2 donor values keep exact UTF-8 text as their permanent identity.  JTD
compiles each text key into two canonicalized Qwen-token addresses:

* beginning-of-sequence encoding;
* ordinary internal-text encoding with leading-space behavior.

Raw Qwen IDs remain the language-model stream.  The frozen Engram projection
is applied only to memory addresses.  A parallel complete-word index supports
offset-aligned lookup when a tokenizer piece fuses a word with punctuation.
Both paths use compact hashes plus complete-key verification. Numerical latent
alignment lives in :mod:`dendritron.joint_transfer`.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

from .retrieval import LongestEngramRouter, MemoryCandidate, ResolvedMemory
from .tokenizer import (
    CanonicalTokenProjection,
    TokenizerContract,
    TokenizerLike,
    boundary_token_ids,
    canonical_token_text,
)


SURFACE_INDEX_SCHEMA_VERSION = 3
JTD_SCHEMA_VERSION = SURFACE_INDEX_SCHEMA_VERSION  # historical import alias
JTD_HASH_PERSON = b"DEND-JTD-v1"
JTD_WORD_HASH_PERSON = b"DEND-WORD-v1"
BankName = Literal["bigrams", "trigrams", "dictionary"]


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def encode_token_tuple(token_ids: Sequence[int]) -> bytes:
    values = tuple(int(value) for value in token_ids)
    if not values:
        raise ValueError("token tuple cannot be empty")
    if min(values) < 0 or max(values) > 0xFFFFFFFF:
        raise ValueError("token IDs must fit unsigned 32-bit storage")
    return struct.pack("<I", len(values)) + struct.pack(
        f"<{len(values)}I",
        *values,
    )


def decode_token_tuple(encoded: bytes) -> tuple[int, ...]:
    if len(encoded) < 8 or len(encoded) % 4:
        raise ValueError("invalid encoded token tuple")
    count = struct.unpack_from("<I", encoded, 0)[0]
    if len(encoded) != 4 + count * 4 or count < 1:
        raise ValueError("encoded token tuple length mismatch")
    return tuple(struct.unpack_from(f"<{count}I", encoded, 4))


def token_tuple_hash(encoded: bytes) -> bytes:
    return hashlib.blake2b(
        encoded,
        digest_size=16,
        person=JTD_HASH_PERSON,
    ).digest()


def encode_word_tuple(words: Sequence[str]) -> bytes:
    values = tuple(str(value) for value in words)
    if not values or any(not value for value in values):
        raise ValueError("word tuple cannot be empty")
    return json.dumps(
        values,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def word_tuple_hash(encoded: bytes) -> bytes:
    return hashlib.blake2b(
        encoded,
        digest_size=16,
        person=JTD_WORD_HASH_PERSON,
    ).digest()


def canonical_surface_words(surface_text: str, word_order: int) -> tuple[str, ...]:
    words = tuple(canonical_token_text(value) for value in surface_text.split())
    if len(words) != word_order or any(not value for value in words):
        raise ValueError(
            f"Expected {word_order} complete words in {surface_text!r}, found {words!r}"
        )
    return words


@dataclass(frozen=True)
class JTDSourceRecord:
    bank_name: BankName
    word_order: int
    row_index: int
    surface_text: str
    frequency: int = 0
    sense_id: str | None = None

    def __post_init__(self) -> None:
        expected = {
            "bigrams": 2,
            "trigrams": 3,
            "dictionary": 1,
        }[self.bank_name]
        if self.word_order != expected:
            raise ValueError(
                f"{self.bank_name} records require word_order={expected}"
            )
        if self.row_index < 0:
            raise ValueError("row_index must be nonnegative")
        if not self.surface_text or self.surface_text != self.surface_text.strip():
            raise ValueError("surface_text must be exact text without outer whitespace")
        if self.bank_name == "dictionary" and not self.sense_id:
            raise ValueError("dictionary records require sense_id")


def iter_stage2_key_records(
    path: Path,
    *,
    bank_name: Literal["bigrams", "trigrams"],
) -> Iterator[JTDSourceRecord]:
    expected_order = 2 if bank_name == "bigrams" else 3
    next_row = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            row_index = int(record.get("row_index", next_row))
            if row_index != next_row:
                raise ValueError(
                    f"{path}:{line_number}: expected row {next_row}, found {row_index}"
                )
            order = int(record.get("n", expected_order))
            if order != expected_order:
                raise ValueError(
                    f"{path}:{line_number}: expected n={expected_order}, found {order}"
                )
            yield JTDSourceRecord(
                bank_name=bank_name,
                word_order=expected_order,
                row_index=row_index,
                surface_text=str(record["text"]),
                frequency=int(record.get("frequency", 0)),
            )
            next_row += 1


def iter_dictionary_records(path: Path) -> Iterator[JTDSourceRecord]:
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT
                senses.sense_row,
                senses.sense_id,
                words.surface
            FROM senses
            JOIN words ON words.word_id = senses.word_id
            ORDER BY senses.sense_row
            """
        )
        next_row = 0
        for row_index, sense_id, surface in rows:
            if int(row_index) != next_row:
                raise ValueError(
                    f"Dictionary sense rows are discontinuous at {row_index}"
                )
            yield JTDSourceRecord(
                bank_name="dictionary",
                word_order=1,
                row_index=int(row_index),
                surface_text=str(surface),
                sense_id=str(sense_id),
            )
            next_row += 1
    finally:
        connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;

        CREATE TABLE token_projection (
            raw_token_id INTEGER PRIMARY KEY CHECK(raw_token_id >= 0),
            canonical_token_id INTEGER NOT NULL CHECK(canonical_token_id >= 0)
        ) WITHOUT ROWID;

        CREATE TABLE addresses (
            boundary_mode TEXT NOT NULL CHECK(boundary_mode IN ('bos', 'internal')),
            token_length INTEGER NOT NULL CHECK(token_length > 0),
            token_hash BLOB NOT NULL,
            token_ids BLOB NOT NULL,
            memory_kind TEXT NOT NULL CHECK(
                memory_kind IN ('donor_engram', 'dictionary_sense')
            ),
            bank_name TEXT NOT NULL CHECK(
                bank_name IN ('bigrams', 'trigrams', 'dictionary')
            ),
            word_order INTEGER NOT NULL CHECK(word_order IN (1, 2, 3)),
            row_index INTEGER NOT NULL CHECK(row_index >= 0),
            surface_text TEXT NOT NULL,
            frequency INTEGER NOT NULL CHECK(frequency >= 0),
            sense_id TEXT,
            PRIMARY KEY(bank_name, row_index, boundary_mode)
        ) WITHOUT ROWID;

        CREATE INDEX address_lookup
        ON addresses(boundary_mode, token_length, token_hash);

        CREATE INDEX address_rows
        ON addresses(bank_name, row_index);

        CREATE TABLE lexical_addresses (
            word_order INTEGER NOT NULL CHECK(word_order IN (1, 2, 3)),
            word_hash BLOB NOT NULL,
            words BLOB NOT NULL,
            memory_kind TEXT NOT NULL CHECK(
                memory_kind IN ('donor_engram', 'dictionary_sense')
            ),
            bank_name TEXT NOT NULL CHECK(
                bank_name IN ('bigrams', 'trigrams', 'dictionary')
            ),
            row_index INTEGER NOT NULL CHECK(row_index >= 0),
            surface_text TEXT NOT NULL,
            frequency INTEGER NOT NULL CHECK(frequency >= 0),
            sense_id TEXT,
            PRIMARY KEY(bank_name, row_index)
        ) WITHOUT ROWID;

        CREATE INDEX lexical_lookup
        ON lexical_addresses(word_order, word_hash);
        """
    )


def compile_surface_index_database(
    records: Iterable[JTDSourceRecord],
    *,
    tokenizer: TokenizerLike,
    tokenizer_contract: TokenizerContract,
    token_projection: CanonicalTokenProjection,
    database_path: Path,
    collision_report_path: Path,
    commit_rows: int = 10_000,
) -> dict[str, Any]:
    """Compile all source rows into one immutable CPU lookup artifact."""
    database_path.parent.mkdir(parents=True, exist_ok=True)
    collision_report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_database = database_path.with_name(database_path.name + ".tmp")
    temporary_collisions = collision_report_path.with_name(
        collision_report_path.name + ".tmp"
    )
    temporary_database.unlink(missing_ok=True)
    temporary_collisions.unlink(missing_ok=True)

    connection = sqlite3.connect(str(temporary_database))
    _create_schema(connection)
    counts = {"bigrams": 0, "trigrams": 0, "dictionary": 0}
    address_rows = 0
    maximum_token_span = 0
    pending: list[tuple[Any, ...]] = []
    lexical_pending: list[tuple[Any, ...]] = []
    try:
        connection.executemany(
            "INSERT INTO token_projection VALUES (?, ?)",
            enumerate(token_projection.raw_to_canonical),
        )
        for source in records:
            variants = boundary_token_ids(tokenizer, source.surface_text)
            for boundary_mode, raw_token_ids in variants.items():
                token_ids = token_projection.project(raw_token_ids)
                encoded = encode_token_tuple(token_ids)
                pending.append(
                    (
                        boundary_mode,
                        len(token_ids),
                        token_tuple_hash(encoded),
                        encoded,
                        (
                            "dictionary_sense"
                            if source.bank_name == "dictionary"
                            else "donor_engram"
                        ),
                        source.bank_name,
                        source.word_order,
                        source.row_index,
                        source.surface_text,
                        source.frequency,
                        source.sense_id,
                    )
                )
                address_rows += 1
                maximum_token_span = max(maximum_token_span, len(token_ids))
            words = canonical_surface_words(source.surface_text, source.word_order)
            encoded_words = encode_word_tuple(words)
            lexical_pending.append(
                (
                    source.word_order,
                    word_tuple_hash(encoded_words),
                    encoded_words,
                    (
                        "dictionary_sense"
                        if source.bank_name == "dictionary"
                        else "donor_engram"
                    ),
                    source.bank_name,
                    source.row_index,
                    source.surface_text,
                    source.frequency,
                    source.sense_id,
                )
            )
            counts[source.bank_name] += 1
            if len(pending) >= commit_rows:
                connection.executemany(
                    """
                    INSERT INTO addresses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    pending,
                )
                connection.executemany(
                    "INSERT INTO lexical_addresses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    lexical_pending,
                )
                connection.commit()
                pending.clear()
                lexical_pending.clear()
        if pending:
            connection.executemany(
                "INSERT INTO addresses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                pending,
            )
            pending.clear()
        if lexical_pending:
            connection.executemany(
                "INSERT INTO lexical_addresses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                lexical_pending,
            )
            lexical_pending.clear()

        metadata = {
            "schema_version": JTD_SCHEMA_VERSION,
            "tokenizer_fingerprint": tokenizer_contract.fingerprint,
            "tokenizer_id": tokenizer_contract.tokenizer_id,
            "tokenizer_resolved_revision": tokenizer_contract.resolved_revision,
            "token_projection": token_projection.to_record(),
            "hash": "blake2b-128-bucket-plus-exact-token-tuple",
            "lexical_hash": "blake2b-128-bucket-plus-exact-word-tuple",
            "maximum_token_span": maximum_token_span,
            "source_rows": counts,
            "address_rows": address_rows,
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                (
                    key,
                    json.dumps(value, ensure_ascii=False, sort_keys=True),
                )
                for key, value in metadata.items()
            ),
        )
        connection.commit()

        collision_groups = connection.execute(
            """
            SELECT
                boundary_mode,
                token_length,
                token_hash,
                token_ids,
                count(*) AS candidate_count,
                count(DISTINCT surface_text) AS surface_count
            FROM addresses
            GROUP BY boundary_mode, token_length, token_hash, token_ids
            HAVING count(DISTINCT surface_text) > 1
                OR count(DISTINCT word_order) > 1
            ORDER BY boundary_mode, token_length, token_hash
            """
        ).fetchall()
        with temporary_collisions.open("w", encoding="utf-8") as handle:
            for (
                boundary_mode,
                token_length,
                token_hash,
                token_ids_blob,
                candidate_count,
                surface_count,
            ) in collision_groups:
                candidates = connection.execute(
                    """
                    SELECT
                        memory_kind,
                        bank_name,
                        word_order,
                        row_index,
                        surface_text,
                        frequency,
                        sense_id
                    FROM addresses
                    WHERE boundary_mode = ?
                      AND token_length = ?
                      AND token_hash = ?
                      AND token_ids = ?
                    ORDER BY
                        word_order DESC,
                        frequency DESC,
                        row_index ASC
                    """,
                    (
                        boundary_mode,
                        token_length,
                        token_hash,
                        token_ids_blob,
                    ),
                ).fetchall()
                report = {
                    "boundary_mode": boundary_mode,
                    "token_ids": list(decode_token_tuple(token_ids_blob)),
                    "candidate_count": int(candidate_count),
                    "surface_count": int(surface_count),
                    "candidates": [
                        {
                            "memory_kind": row[0],
                            "bank_name": row[1],
                            "word_order": int(row[2]),
                            "row_index": int(row[3]),
                            "surface_text": row[4],
                            "frequency": int(row[5]),
                            "sense_id": row[6],
                        }
                        for row in candidates
                    ],
                }
                handle.write(
                    json.dumps(
                        report,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        cryptographic_collisions = int(
            connection.execute(
                """
                SELECT count(*)
                FROM (
                    SELECT boundary_mode, token_length, token_hash
                    FROM addresses
                    GROUP BY boundary_mode, token_length, token_hash
                    HAVING count(DISTINCT token_ids) > 1
                )
                """
            ).fetchone()[0]
        )
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(
                f"Surface-index SQLite integrity check failed: {integrity}"
            )
    finally:
        connection.close()

    os.replace(temporary_database, database_path)
    os.replace(temporary_collisions, collision_report_path)
    return {
        "schema_version": JTD_SCHEMA_VERSION,
        "database": {
            "path": str(database_path),
            "bytes": database_path.stat().st_size,
            "sha256": file_sha256(database_path),
        },
        "collision_report": {
            "path": str(collision_report_path),
            "bytes": collision_report_path.stat().st_size,
            "sha256": file_sha256(collision_report_path),
            "address_collision_groups": len(collision_groups),
            "cryptographic_hash_collisions": cryptographic_collisions,
        },
        "source_rows": counts,
        "address_rows": address_rows,
        "maximum_token_span": maximum_token_span,
        "tokenizer_fingerprint": tokenizer_contract.fingerprint,
        "token_projection": token_projection.to_record(),
    }


class SurfaceMemoryIndex:
    """Read-only collision-safe lookup over a compiled surface database."""

    def __init__(
        self,
        database_path: Path,
        *,
        expected_tokenizer_fingerprint: str | None = None,
        expected_projection_fingerprint: str | None = None,
    ) -> None:
        self.database_path = Path(database_path)
        uri = f"file:{self.database_path.resolve()}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)
        metadata = {
            key: json.loads(value)
            for key, value in self._connection.execute(
                "SELECT key, value FROM metadata"
            )
        }
        if int(metadata.get("schema_version", -1)) != JTD_SCHEMA_VERSION:
            self.close()
            raise ValueError("Unsupported surface-index database schema")
        if (
            expected_tokenizer_fingerprint is not None
            and metadata.get("tokenizer_fingerprint")
            != expected_tokenizer_fingerprint
        ):
            self.close()
            raise ValueError("Surface-index/tokenizer fingerprint mismatch")
        self.metadata = metadata
        self.maximum_token_span = int(metadata["maximum_token_span"])
        projection_record = metadata.get("token_projection")
        if not isinstance(projection_record, Mapping):
            self.close()
            raise ValueError("Surface index has no canonical token projection")
        if (
            expected_projection_fingerprint is not None
            and projection_record.get("fingerprint")
            != expected_projection_fingerprint
        ):
            self.close()
            raise ValueError("Surface-index/token-projection fingerprint mismatch")
        rows = self._connection.execute(
            "SELECT raw_token_id, canonical_token_id "
            "FROM token_projection ORDER BY raw_token_id"
        ).fetchall()
        self.raw_to_canonical = tuple(int(row[1]) for row in rows)
        if any(int(row[0]) != index for index, row in enumerate(rows)):
            self.close()
            raise ValueError("Surface token projection contains discontinuous raw IDs")
        if len(self.raw_to_canonical) != int(projection_record["raw_vocab_size"]):
            self.close()
            raise ValueError("Surface token projection size mismatch")
        if self.raw_to_canonical and (
            min(self.raw_to_canonical) < 0
            or max(self.raw_to_canonical)
            >= int(projection_record["effective_vocab_size"])
        ):
            self.close()
            raise ValueError("Surface token projection contains an invalid canonical ID")
        encoded_projection = struct.pack(
            f"<{len(self.raw_to_canonical)}I",
            *self.raw_to_canonical,
        )
        actual_projection_fingerprint = hashlib.sha256(
            str(projection_record["algorithm"]).encode("utf-8")
            + b"\0"
            + encoded_projection
        ).hexdigest()
        if actual_projection_fingerprint != str(projection_record["fingerprint"]):
            self.close()
            raise ValueError("Surface token projection fingerprint mismatch")

    def project_token_ids(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        projected: list[int] = []
        for raw_value in token_ids:
            raw_id = int(raw_value)
            if not 0 <= raw_id < len(self.raw_to_canonical):
                raise ValueError(f"Raw token ID {raw_id} falls outside the Qwen vocabulary")
            projected.append(self.raw_to_canonical[raw_id])
        return tuple(projected)

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self) -> "SurfaceMemoryIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _candidate_rows(
        self,
        token_ids: Sequence[int],
        end_position: int,
    ) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        largest_span = min(self.maximum_token_span, end_position + 1)
        for token_length in range(1, largest_span + 1):
            start_position = end_position - token_length + 1
            boundary_mode = "bos" if start_position == 0 else "internal"
            suffix = tuple(int(value) for value in token_ids[start_position : end_position + 1])
            encoded = encode_token_tuple(suffix)
            rows = self._connection.execute(
                """
                SELECT
                    memory_kind,
                    word_order,
                    row_index,
                    surface_text,
                    frequency,
                    sense_id,
                    token_ids
                FROM addresses
                WHERE boundary_mode = ?
                  AND token_length = ?
                  AND token_hash = ?
                """,
                (
                    boundary_mode,
                    token_length,
                    token_tuple_hash(encoded),
                ),
            )
            for (
                memory_kind,
                word_order,
                row_index,
                surface_text,
                frequency,
                sense_id,
                stored_token_ids,
            ) in rows:
                if stored_token_ids != encoded:
                    continue
                candidates.append(
                    MemoryCandidate(
                        kind=memory_kind,
                        word_order=int(word_order),
                        row_index=int(row_index),
                        surface_text=str(surface_text),
                        recipient_ids=suffix,
                        boundary_mode=boundary_mode,
                        frequency=int(frequency),
                        sense_id=(
                            None if sense_id is None else str(sense_id)
                        ),
                    )
                )
        return candidates

    def _lexical_candidates(
        self,
        words: Sequence[str],
        start_position: int,
        end_position: int,
    ) -> list[MemoryCandidate]:
        normalized = tuple(
            canonical_token_text(value)
            for value in words[start_position : end_position + 1]
        )
        encoded = encode_word_tuple(normalized)
        rows = self._connection.execute(
            """
            SELECT
                memory_kind,
                word_order,
                row_index,
                surface_text,
                frequency,
                sense_id,
                words
            FROM lexical_addresses
            WHERE word_order = ? AND word_hash = ?
            ORDER BY frequency DESC, row_index ASC
            """,
            (len(normalized), word_tuple_hash(encoded)),
        )
        boundary_mode = "bos" if start_position == 0 else "internal"
        return [
            MemoryCandidate(
                kind=str(memory_kind),
                word_order=int(word_order),
                row_index=int(row_index),
                surface_text=str(surface_text),
                recipient_ids=normalized,
                boundary_mode=boundary_mode,
                frequency=int(frequency),
                sense_id=None if sense_id is None else str(sense_id),
            )
            for (
                memory_kind,
                word_order,
                row_index,
                surface_text,
                frequency,
                sense_id,
                stored_words,
            ) in rows
            if stored_words == encoded
        ]

    def resolve_words(
        self,
        words: Sequence[str],
        boundary_before: Sequence[bool],
        end_position: int,
        *,
        include_decomposition: bool = False,
    ) -> ResolvedMemory | None:
        """Resolve word Engrams without allowing punctuation crossings."""

        if len(words) != len(boundary_before):
            raise ValueError("words and boundary_before must have equal length")
        if not 0 <= end_position < len(words):
            raise IndexError("end_position falls outside words")
        segment_start = 0
        for position in range(end_position, -1, -1):
            if boundary_before[position]:
                segment_start = position
                break

        matches: dict[int, list[MemoryCandidate]] = {}
        for order in (3, 2, 1):
            start = end_position - order + 1
            if start < segment_start:
                continue
            candidates = self._lexical_candidates(words, start, end_position)
            if candidates:
                matches[order] = candidates
        if not matches:
            return None

        selected_order = max(matches)
        selected_candidates = matches[selected_order]
        if selected_order in {2, 3}:
            selected = (selected_candidates[0],)
        else:
            selected = tuple(selected_candidates)
        decomposition: tuple[MemoryCandidate, ...] = ()
        if include_decomposition:
            decomposition = tuple(
                candidate
                for order in sorted(matches, reverse=True)
                if order < selected_order
                for candidate in matches[order]
            )
        return ResolvedMemory(
            end_position=end_position,
            start_position=end_position - selected_order + 1,
            word_order=selected_order,
            selected=selected,
            decomposition_candidates=decomposition,
        )

    def resolve(
        self,
        token_ids: Sequence[int],
        end_position: int,
        *,
        include_decomposition: bool = False,
    ) -> ResolvedMemory | None:
        projected_ids = self.project_token_ids(token_ids)
        candidates = self._candidate_rows(projected_ids, end_position)
        if not candidates:
            return None
        router = LongestEngramRouter(candidates)
        return router.resolve(
            projected_ids,
            end_position,
            include_decomposition=include_decomposition,
        )

    def resolve_sequence(
        self,
        token_ids: Sequence[int],
        *,
        include_decomposition: bool = False,
    ) -> tuple[ResolvedMemory | None, ...]:
        return tuple(
            self.resolve(
                token_ids,
                end_position,
                include_decomposition=include_decomposition,
            )
            for end_position in range(len(token_ids))
        )


# Compatibility names for pre-v1.2 callers. New code uses the surface names;
# latent alignment is implemented by JointTransferDomain in joint_transfer.py.
compile_jtd_database = compile_surface_index_database
JTDIndex = SurfaceMemoryIndex
