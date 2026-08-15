"""Stable schemas and helpers for Dendritron's dictionary knowledge bank.

The dictionary is sense-addressed rather than word-addressed. A polysemous
surface such as ``bark`` therefore owns several immutable sense rows. Every
sense preserves the ordered words in its definition and receives one donor
layer-2 readout vector. The shallow donor view keeps the definition lexical
and compositional; the ordered word links preserve its explicit constituents.

The full 2,048D vector and definition-word links form the canonical knowledge
record. They live in a CPU/disk lookup bank and are fetched sparsely by sense.
The separate Universal/Shared-LoRA subspace is learned from task updates in
weight space.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = 1
DEFINITION_READOUT_MARKER = "\nConcept:"


def normalize_word(surface: str) -> str:
    """Return the permanent case-insensitive identity for one word."""
    return unicodedata.normalize("NFKC", surface).casefold().strip()


def canonical_definition_text(
    definition: str,
    marker: str = DEFINITION_READOUT_MARKER,
) -> str:
    """Build the donor input whose final marker token becomes the readout."""
    cleaned = " ".join(str(definition).split())
    if not cleaned:
        raise ValueError("A definition must contain text")
    if not marker:
        raise ValueError("The definition readout marker must contain text")
    return cleaned + marker


def iter_definition_words(text: str) -> Iterator[str]:
    """Yield normalized Unicode word units in their original order."""
    cursor = 0
    while cursor < len(text):
        category = unicodedata.category(text[cursor])[:1]
        if category not in {"L", "N"}:
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(text):
            character = text[cursor]
            category = unicodedata.category(character)[:1]
            if category in {"L", "M", "N"}:
                cursor += 1
                continue
            if (
                character in {"'", "’", "-", "\u2011"}
                and cursor + 1 < len(text)
                and unicodedata.category(text[cursor + 1])[:1] in {"L", "N"}
            ):
                cursor += 1
                continue
            break
        yield text[start:cursor]


def stable_identifier(*parts: str, prefix: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8", errors="surrogatepass")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:24]}"


@dataclass(frozen=True)
class DictionaryWord:
    word_id: int
    surface: str
    normalized: str
    in_ngram_vocabulary: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DefinitionSense:
    sense_row: int
    sense_id: str
    word_id: int
    surface: str
    normalized: str
    part_of_speech: str
    definition: str
    source: str
    source_sense_key: str
    definition_words: tuple[str, ...]
    examples: tuple[str, ...] = ()

    @property
    def donor_text(self) -> str:
        return canonical_definition_text(self.definition)

    def to_record(self, include_donor_text: bool = True) -> dict[str, Any]:
        record = asdict(self)
        record["definition_words"] = list(self.definition_words)
        record["examples"] = list(self.examples)
        if include_donor_text:
            record["donor_text"] = self.donor_text
        return record


def _first_present(record: Mapping[str, Any], names: Iterable[str]) -> str:
    for name in names:
        value = record.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_source_record(
    record: Mapping[str, Any],
    *,
    default_source: str,
    ordinal: int,
) -> dict[str, Any]:
    """Normalize WordNet/Wiktionary/custom JSONL into one sense candidate."""
    surface = _first_present(record, ("surface", "word", "lemma", "headword"))
    definition = _first_present(record, ("definition", "gloss", "meaning"))
    if not surface:
        raise ValueError(f"Definition record {ordinal} has no word or lemma")
    if not definition:
        raise ValueError(f"Definition record {ordinal} has no definition or gloss")

    source = _first_present(record, ("source",)) or default_source
    pos = _first_present(record, ("part_of_speech", "pos"))
    source_key = _first_present(
        record,
        ("source_sense_key", "sense_key", "synset_id", "id"),
    )
    if not source_key:
        source_key = stable_identifier(
            normalize_word(surface),
            pos,
            definition,
            prefix="sense",
        )

    examples_value = record.get("examples", ())
    if isinstance(examples_value, str):
        examples = (examples_value.strip(),) if examples_value.strip() else ()
    else:
        examples = tuple(
            str(value).strip()
            for value in (examples_value or ())
            if str(value).strip()
        )

    return {
        "surface": unicodedata.normalize("NFKC", surface).strip(),
        "normalized": normalize_word(surface),
        "part_of_speech": pos,
        "definition": " ".join(definition.split()),
        "source": source,
        "source_sense_key": source_key,
        "examples": examples,
    }


def build_dictionary_records(
    candidates: Iterable[Mapping[str, Any]],
    *,
    ngram_words: Iterable[str] = (),
    default_source: str = "custom",
) -> tuple[list[DictionaryWord], list[DefinitionSense], dict[str, Any]]:
    """Deduplicate candidates and assign stable word and sense row numbers."""
    ngram_surface_by_normalized: dict[str, str] = {}
    for value in ngram_words:
        ngram_surface_by_normalized.setdefault(normalize_word(value), value)
    ngram_normalized = set(ngram_surface_by_normalized)
    normalized_candidates: list[dict[str, Any]] = []
    seen_senses: set[tuple[str, str, str]] = set()

    for ordinal, candidate in enumerate(candidates, start=1):
        normalized = normalize_source_record(
            candidate,
            default_source=default_source,
            ordinal=ordinal,
        )
        identity = (
            normalized["source"],
            normalized["source_sense_key"],
            normalized["normalized"],
        )
        if identity in seen_senses:
            continue
        seen_senses.add(identity)
        normalized_candidates.append(normalized)

    all_words = {item["normalized"] for item in normalized_candidates}
    definition_word_surfaces: dict[str, str] = {}
    for item in normalized_candidates:
        for definition_word in iter_definition_words(item["definition"]):
            normalized_definition_word = normalize_word(definition_word)
            all_words.add(normalized_definition_word)
            definition_word_surfaces.setdefault(
                normalized_definition_word,
                definition_word,
            )
    all_words.update(ngram_normalized)
    canonical_surface: dict[str, str] = {
        item["normalized"]: item["surface"] for item in normalized_candidates
    }
    for normalized, surface in ngram_surface_by_normalized.items():
        canonical_surface.setdefault(normalized, surface)
    for normalized, surface in definition_word_surfaces.items():
        canonical_surface.setdefault(normalized, surface)

    words: list[DictionaryWord] = []
    word_ids: dict[str, int] = {}
    for word_id, normalized in enumerate(sorted(all_words)):
        word_ids[normalized] = word_id
        words.append(
            DictionaryWord(
                word_id=word_id,
                surface=canonical_surface[normalized],
                normalized=normalized,
                in_ngram_vocabulary=normalized in ngram_normalized,
            )
        )

    sorted_candidates = sorted(
        normalized_candidates,
        key=lambda item: (
            word_ids[item["normalized"]],
            item["source"],
            item["source_sense_key"],
            item["definition"],
        ),
    )
    senses: list[DefinitionSense] = []
    for sense_row, item in enumerate(sorted_candidates):
        senses.append(
            DefinitionSense(
                sense_row=sense_row,
                sense_id=stable_identifier(
                    item["source"],
                    item["source_sense_key"],
                    item["normalized"],
                    prefix="sense",
                ),
                word_id=word_ids[item["normalized"]],
                surface=item["surface"],
                normalized=item["normalized"],
                part_of_speech=item["part_of_speech"],
                definition=item["definition"],
                source=item["source"],
                source_sense_key=item["source_sense_key"],
                definition_words=tuple(iter_definition_words(item["definition"])),
                examples=item["examples"],
            )
        )

    words_with_senses = {sense.word_id for sense in senses}
    ngram_word_ids = {word.word_id for word in words if word.in_ngram_vocabulary}
    definition_word_ids = {
        word_ids[normalized] for normalized in definition_word_surfaces
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "word_rows": len(words),
        "sense_rows": len(senses),
        "ngram_vocabulary_words": len(ngram_word_ids),
        "ngram_words_with_definition": len(ngram_word_ids & words_with_senses),
        "ngram_words_without_definition": len(ngram_word_ids - words_with_senses),
        "definition_word_nodes": len(definition_word_ids),
        "definition_word_nodes_with_senses": len(
            definition_word_ids & words_with_senses
        ),
        "definition_word_nodes_as_links_only": len(
            definition_word_ids - words_with_senses
        ),
        "definition_word_edges": sum(
            len(sense.definition_words) for sense in senses
        ),
        "definition_readout_marker": DEFINITION_READOUT_MARKER,
        "definition_vector_layer": 2,
        "definition_structure": "ordered_word_links_plus_layer02_readout",
        "runtime_storage": "cpu_disk_sparse_lookup",
        "dendritron_cuda_required": False,
    }
    return words, senses, report


def write_jsonl(records: Iterable[Mapping[str, Any]], handle: Any) -> None:
    for record in records:
        handle.write(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def read_jsonl(handle: Any) -> Iterator[dict[str, Any]]:
    for line_number, raw_line in enumerate(handle, start=1):
        if not raw_line.strip():
            continue
        value = json.loads(raw_line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL row {line_number} must be an object")
        yield value
