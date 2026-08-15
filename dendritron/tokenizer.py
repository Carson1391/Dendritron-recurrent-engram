"""Qwen tokenizer and canonical conditional-memory address contracts.

Dendritron uses one exact Qwen tokenizer revision for the raw language-model
stream.  Raw IDs feed the learned input embeddings and the tied output
vocabulary.  A frozen surjective projection maps a parallel copy of those IDs
to canonical IDs for JTD and Hash-Engram addressing only.

The projection follows the official Engram demonstration algorithm:
NFKC -> NFD -> strip accents -> lowercase -> collapse whitespace -> strip.
Whitespace-only pieces canonicalize to one space.  Tokens that decode through
the Unicode replacement character use their raw tokenizer piece as the key.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


TOKENIZER_SCHEMA_VERSION = 1
PROJECTION_SCHEMA_VERSION = 1
LOCKED_QWEN_TOKENIZER_ID = "Qwen/Qwen3.6-35B-A3B"
SUPPORTED_RECIPIENT_TOKENIZERS = (LOCKED_QWEN_TOKENIZER_ID,)
LOCKED_ADD_SPECIAL_TOKENS = False
BOUNDARY_SAMPLE_TEXTS = ("bark", "tree bark", "rough tree bark")
CANONICAL_PROJECTION_ALGORITHM = (
    "engram_nfkc_nfd_strip_accents_lower_whitespace_v1"
)
_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")
INTERNAL_WORD_CONNECTORS = frozenset(("'", "’", "-", "\u2011"))


class TokenizerLike(Protocol):
    """Small interface used by the CPU compiler and synthetic tests."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> Sequence[int]: ...


class ProjectionTokenizerLike(TokenizerLike, Protocol):
    def __len__(self) -> int: ...

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...

    def convert_ids_to_tokens(self, token_id: int) -> str: ...


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_token_text(text: str) -> str:
    """Apply Engram's published token-text normalization exactly.

    The official implementation protects a whitespace-only token from the
    final strip operation.  This equivalent implementation returns one plain
    space for every nonempty run containing only spaces, tabs, CR, or LF.
    """

    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = unicodedata.normalize("NFD", normalized)
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    normalized = normalized.lower()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    if normalized == " ":
        return " "
    return normalized.strip()


def _decode_one_token(tokenizer: ProjectionTokenizerLike, token_id: int) -> str:
    try:
        return str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode([token_id], skip_special_tokens=False))


@dataclass(frozen=True)
class CanonicalTokenProjection:
    """Frozen raw-Qwen-ID to canonical-memory-ID projection."""

    raw_to_canonical: tuple[int, ...]
    canonical_keys: tuple[str, ...]
    algorithm: str
    fingerprint: str

    @property
    def raw_vocab_size(self) -> int:
        return len(self.raw_to_canonical)

    @property
    def effective_vocab_size(self) -> int:
        return len(self.canonical_keys)

    @property
    def reduction_fraction(self) -> float:
        if not self.raw_vocab_size:
            return 0.0
        return 1.0 - self.effective_vocab_size / self.raw_vocab_size

    @property
    def reduction_percent(self) -> float:
        return 100.0 * self.reduction_fraction

    def project(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        projected: list[int] = []
        for raw_value in token_ids:
            raw_id = int(raw_value)
            if raw_id < 0:
                projected.append(raw_id)
                continue
            if raw_id >= self.raw_vocab_size:
                raise ValueError(
                    f"Raw token ID {raw_id} exceeds projection vocabulary "
                    f"{self.raw_vocab_size}"
                )
            projected.append(self.raw_to_canonical[raw_id])
        return tuple(projected)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": PROJECTION_SCHEMA_VERSION,
            "algorithm": self.algorithm,
            "raw_vocab_size": self.raw_vocab_size,
            "effective_vocab_size": self.effective_vocab_size,
            "reduction_fraction": self.reduction_fraction,
            "reduction_percent": self.reduction_percent,
            "fingerprint": self.fingerprint,
        }


def build_canonical_token_projection(
    tokenizer: ProjectionTokenizerLike,
) -> CanonicalTokenProjection:
    """Precompute Engram's surjective projection for the locked tokenizer."""

    try:
        vocabulary_size = int(len(tokenizer))
    except TypeError:
        vocabulary_size = int(getattr(tokenizer, "vocab_size"))
    if vocabulary_size < 1:
        raise ValueError("Tokenizer vocabulary must be nonempty")

    key_to_id: dict[str, int] = {}
    canonical_keys: list[str] = []
    raw_to_canonical: list[int] = []
    for token_id in range(vocabulary_size):
        decoded = _decode_one_token(tokenizer, token_id)
        if "\ufffd" in decoded:
            key = str(tokenizer.convert_ids_to_tokens(token_id))
        else:
            normalized = canonical_token_text(decoded)
            key = normalized if normalized else decoded
        canonical_id = key_to_id.get(key)
        if canonical_id is None:
            canonical_id = len(canonical_keys)
            key_to_id[key] = canonical_id
            canonical_keys.append(key)
        raw_to_canonical.append(canonical_id)

    encoded = struct.pack(
        f"<{len(raw_to_canonical)}I",
        *raw_to_canonical,
    )
    fingerprint = hashlib.sha256(
        CANONICAL_PROJECTION_ALGORITHM.encode("utf-8") + b"\0" + encoded
    ).hexdigest()
    return CanonicalTokenProjection(
        raw_to_canonical=tuple(raw_to_canonical),
        canonical_keys=tuple(canonical_keys),
        algorithm=CANONICAL_PROJECTION_ALGORITHM,
        fingerprint=fingerprint,
    )


def _is_word_start(character: str) -> bool:
    return unicodedata.category(character)[:1] in {"L", "N"}


def _is_word_body(character: str) -> bool:
    return unicodedata.category(character)[:1] in {"L", "M", "N"}


def iter_word_spans(text: str):
    """Yield complete Unicode words with internal apostrophes and hyphens."""

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


def complete_word_segments(text: str) -> tuple[tuple[str, ...], ...]:
    """Split text into complete-word runs separated by punctuation/symbols."""

    segments: list[list[str]] = []
    current: list[str] = []
    previous_end = 0
    for char_start, char_end in iter_word_spans(text):
        gap = text[previous_end:char_start]
        if current and any(not character.isspace() for character in gap):
            segments.append(current)
            current = []
        current.append(text[char_start:char_end])
        previous_end = char_end
    if current:
        segments.append(current)
    return tuple(tuple(segment) for segment in segments)


def _flatten_tokenizer_field(value: Any, field: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"Expected one {field} sequence")
        value = value[0]
    return list(value)


@dataclass(frozen=True)
class AlignedWord:
    surface: str
    char_start: int
    char_end: int
    token_positions: tuple[int, ...]
    boundary_before: bool


@dataclass(frozen=True)
class AlignedQwenInput:
    text: str
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    words: tuple[AlignedWord, ...]


def align_qwen_input(tokenizer: Any, text: str) -> AlignedQwenInput:
    """Align complete word units to the actual full-input Qwen tokenization.

    Any non-whitespace material between adjacent complete words starts a new
    frozen word-Engram segment.  Thus commas, periods, quotes, parentheses,
    slashes, and operators block a bigram/trigram crossing.  Apostrophes and
    hyphens internal to a word remain part of that word.
    """

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    token_ids = tuple(
        int(value)
        for value in _flatten_tokenizer_field(encoded["input_ids"], "input")
    )
    offsets = tuple(
        (int(start), int(end))
        for start, end in _flatten_tokenizer_field(
            encoded["offset_mapping"],
            "offset",
        )
    )
    if len(token_ids) != len(offsets):
        raise RuntimeError("Qwen tokenizer returned mismatched IDs and offsets")

    words: list[AlignedWord] = []
    previous_end = 0
    for word_index, (char_start, char_end) in enumerate(iter_word_spans(text)):
        positions = tuple(
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > char_start and token_start < char_end
        )
        if not positions:
            raise RuntimeError(
                f"No Qwen token overlaps word {text[char_start:char_end]!r}"
            )
        gap = text[previous_end:char_start]
        boundary_before = bool(
            word_index > 0 and any(not character.isspace() for character in gap)
        )
        words.append(
            AlignedWord(
                surface=text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                token_positions=positions,
                boundary_before=boundary_before,
            )
        )
        previous_end = char_end
    return AlignedQwenInput(
        text=text,
        input_ids=token_ids,
        offsets=offsets,
        words=tuple(words),
    )


def encode_without_special_tokens(
    tokenizer: TokenizerLike,
    text: str,
) -> tuple[int, ...]:
    ids = tuple(
        int(value)
        for value in tokenizer.encode(
            text,
            add_special_tokens=LOCKED_ADD_SPECIAL_TOKENS,
        )
    )
    if not ids:
        raise ValueError(f"Tokenizer produced an empty address for {text!r}")
    if min(ids) < 0:
        raise ValueError("Tokenizer IDs must be nonnegative")
    return ids


def boundary_token_ids(
    tokenizer: TokenizerLike,
    surface_text: str,
) -> dict[str, tuple[int, ...]]:
    """Compile BOS and ordinary internal-text addresses for one surface key."""
    if not surface_text or surface_text != surface_text.strip():
        raise ValueError(
            "surface_text must be nonempty exact text without outer whitespace"
        )
    return {
        "bos": encode_without_special_tokens(tokenizer, surface_text),
        "internal": encode_without_special_tokens(tokenizer, " " + surface_text),
    }


def tokenizer_source_from_stage2_manifest(
    manifest: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Return tokenizer ID, requested revision, and resolved revision."""
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Stage-2 manifest has no model contract")
    tokenizer_id = str(model.get("id", "")).strip()
    requested_revision = str(model.get("requested_revision", "")).strip()
    resolved_revision = str(model.get("resolved_revision", "")).strip()
    if tokenizer_id != LOCKED_QWEN_TOKENIZER_ID:
        raise ValueError(
            "Stage-2 model ID differs from the locked Qwen tokenizer: "
            f"{tokenizer_id!r}"
        )
    if not requested_revision or not resolved_revision:
        raise ValueError("Stage-2 manifest must record both model revisions")
    return tokenizer_id, requested_revision, resolved_revision


def _special_token_ids(tokenizer: Any) -> dict[str, int | None]:
    names = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "mask_token_id",
    )
    result: dict[str, int | None] = {}
    for name in names:
        value = getattr(tokenizer, name, None)
        result[name] = None if value is None else int(value)
    return result


@dataclass(frozen=True)
class TokenizerContract:
    schema_version: int
    tokenizer_id: str
    requested_revision: str
    resolved_revision: str
    tokenizer_class: str
    is_fast: bool
    vocab_size: int
    add_special_tokens: bool
    padding_side: str
    truncation_side: str
    special_token_ids: dict[str, int | None]
    boundary_samples: dict[str, dict[str, list[int]]]
    snapshot_files: tuple[dict[str, Any], ...]
    fingerprint: str

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["snapshot_files"] = list(self.snapshot_files)
        return record


def build_tokenizer_contract(
    tokenizer: TokenizerLike,
    *,
    tokenizer_id: str,
    requested_revision: str,
    resolved_revision: str,
    snapshot_root: Path | None = None,
) -> TokenizerContract:
    if tokenizer_id not in SUPPORTED_RECIPIENT_TOKENIZERS:
        raise ValueError(
            f"Unsupported recipient tokenizer: {tokenizer_id}"
        )
    boundary_samples = {
        text: {
            mode: list(ids)
            for mode, ids in boundary_token_ids(tokenizer, text).items()
        }
        for text in BOUNDARY_SAMPLE_TEXTS
    }
    snapshot_files: list[dict[str, Any]] = []
    if snapshot_root is not None:
        for path in sorted(
            candidate
            for candidate in snapshot_root.rglob("*")
            if candidate.is_file()
        ):
            snapshot_files.append(
                {
                    "path": str(path.relative_to(snapshot_root)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )

    try:
        vocab_size_value = len(tokenizer)  # type: ignore[arg-type]
    except TypeError:
        vocab_size_value = getattr(tokenizer, "vocab_size", None)
    if vocab_size_value is None:
        raise ValueError("Tokenizer does not expose a vocabulary size")
    base: dict[str, Any] = {
        "schema_version": TOKENIZER_SCHEMA_VERSION,
        "tokenizer_id": tokenizer_id,
        "requested_revision": requested_revision,
        "resolved_revision": resolved_revision,
        "tokenizer_class": type(tokenizer).__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": int(vocab_size_value),
        "add_special_tokens": LOCKED_ADD_SPECIAL_TOKENS,
        "padding_side": str(getattr(tokenizer, "padding_side", "right")),
        "truncation_side": str(getattr(tokenizer, "truncation_side", "right")),
        "special_token_ids": _special_token_ids(tokenizer),
        "boundary_samples": boundary_samples,
        "snapshot_files": tuple(snapshot_files),
    }
    return TokenizerContract(
        **base,
        fingerprint=json_fingerprint(base),
    )


def validate_tokenizer_contract(
    contract: Mapping[str, Any],
    *,
    snapshot_root: Path | None = None,
) -> None:
    if int(contract.get("schema_version", -1)) != TOKENIZER_SCHEMA_VERSION:
        raise ValueError("Unsupported tokenizer contract schema")
    if contract.get("tokenizer_id") not in SUPPORTED_RECIPIENT_TOKENIZERS:
        raise ValueError("Tokenizer contract uses an unsupported tokenizer")
    if bool(contract.get("add_special_tokens", True)):
        raise ValueError("Dendritron token addresses require add_special_tokens=False")

    expected = dict(contract)
    fingerprint = str(expected.pop("fingerprint", ""))
    if json_fingerprint(expected) != fingerprint:
        raise ValueError("Tokenizer contract fingerprint mismatch")

    if snapshot_root is not None:
        for artifact in contract.get("snapshot_files", ()):
            path = snapshot_root / str(artifact["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.stat().st_size != int(artifact["bytes"]):
                raise ValueError(f"Tokenizer file size mismatch: {path}")
            if file_sha256(path) != str(artifact["sha256"]):
                raise ValueError(f"Tokenizer file hash mismatch: {path}")
