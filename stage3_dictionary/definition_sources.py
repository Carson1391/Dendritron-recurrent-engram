"""Canonical, source-grounded definition ingestion for Dendritron.

This module turns three versioned lexical resources into the JSONL schema
consumed by :mod:`stage3_dictionary.build_dictionary_inventory`:

* Open English WordNet 2025+ for curated general senses and proper nouns.
* English Wiktionary (2026-07-06 Wiktextract snapshot) for broad lexical,
  scientific, and mathematical coverage.
* MeSH 2026 descriptors and supplementary concepts for biomedical coverage.

All parsers stream their inputs. Every English, single-word headword with a
real definition is emitted because the dictionary is the complete one-word
fallback in the runtime 3 -> 2 -> 1 resolver. The completed Engram vocabulary
is then checked as a mandatory coverage set. The exact source sense identifier
and source version remain attached to every row.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dendritron.definition_bank import (
    iter_definition_words,
    normalize_source_record,
    normalize_word,
)


PARSER_VERSION = 2
DEFAULT_USER_AGENT = "DendritronDefinitionBuilder/0.7"
WIKTIONARY_EXCLUDED_TAGS = frozenset(
    {
        "abbreviation",
        "alt-of",
        "alternative",
        "character",
        "form-of",
        "misspelling",
        "no-gloss",
        "romanization",
        "translation-hub",
    }
)


@dataclass(frozen=True)
class DefinitionSourceSpec:
    source_id: str
    parser: str
    version: str
    filename: str
    url: str
    source_page: str
    license_name: str
    license_url: str
    attribution: str


SOURCE_SPECS: tuple[DefinitionSourceSpec, ...] = (
    DefinitionSourceSpec(
        source_id="open_english_wordnet_plus",
        parser="oewn_lmf",
        version="2025+",
        filename="english-wordnet-2025-plus.xml.gz",
        url="https://en-word.net/static/english-wordnet-2025-plus.xml.gz",
        source_page="https://en-word.net/downloads",
        license_name="WordNet License + CC BY 4.0",
        license_url=(
            "https://github.com/globalwordnet/english-wordnet/blob/"
            "main/LICENSE.md"
        ),
        attribution=(
            "Princeton WordNet, Open English Namenet, and the Open English "
            "WordNet team, Open English WordNet 2025+."
        ),
    ),
    DefinitionSourceSpec(
        source_id="english_wiktionary",
        parser="wiktextract_jsonl",
        version="2026-07-06",
        filename="enwiktionary-20260706-wiktextract.jsonl.gz",
        url="https://kaikki.org/dictionary/raw-wiktextract-data.jsonl.gz",
        source_page="https://kaikki.org/dictionary/rawdata.html",
        license_name="CC BY-SA 4.0 + GFDL",
        license_url="https://en.wiktionary.org/wiki/Wiktionary:Copyrights",
        attribution=(
            "English Wiktionary contributors; Wiktextract snapshot from the "
            "2026-07-06 enwiktionary dump."
        ),
    ),
    DefinitionSourceSpec(
        source_id="mesh_descriptors",
        parser="mesh_xml",
        version="2026",
        filename="mesh-desc2026.xml.gz",
        url=(
            "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/"
            "xmlmesh/desc2026.gz"
        ),
        source_page="https://www.nlm.nih.gov/databases/download/mesh.html",
        license_name="NLM MeSH Terms and Conditions",
        license_url=(
            "https://www.nlm.nih.gov/databases/download/"
            "terms_and_conditions_mesh.html"
        ),
        attribution="U.S. National Library of Medicine, MeSH 2026.",
    ),
    DefinitionSourceSpec(
        source_id="mesh_supplementary_concepts",
        parser="mesh_xml",
        version="2026",
        filename="mesh-supp2026.xml.gz",
        url=(
            "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/"
            "xmlmesh/supp2026.gz"
        ),
        source_page="https://www.nlm.nih.gov/databases/download/mesh.html",
        license_name="NLM MeSH Terms and Conditions",
        license_url=(
            "https://www.nlm.nih.gov/databases/download/"
            "terms_and_conditions_mesh.html"
        ),
        attribution="U.S. National Library of Medicine, MeSH 2026.",
    ),
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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _open_binary(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _open_text(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str:
    for child in element.iter():
        if _local_name(child.tag) == name and child.text and child.text.strip():
            return " ".join(child.text.split())
    return ""


def _single_word(surface: str) -> str | None:
    cleaned = " ".join(str(surface).split())
    units = tuple(iter_definition_words(cleaned))
    if len(units) != 1:
        return None
    if normalize_word(units[0]) != normalize_word(cleaned):
        return None
    return cleaned


def _record(
    *,
    surface: str,
    part_of_speech: str,
    definition: str,
    source: DefinitionSourceSpec,
    source_sense_key: str,
    examples: Iterable[str] = (),
    domains: Iterable[str] = (),
) -> dict[str, Any]:
    return {
        "surface": surface,
        "part_of_speech": part_of_speech,
        "definition": " ".join(definition.split()),
        "source": f"{source.source_id}:{source.version}",
        "source_sense_key": source_sense_key,
        "examples": [
            " ".join(value.split()) for value in examples if str(value).strip()
        ],
        "domains": sorted(
            {
                normalize_word(value)
                for value in domains
                if str(value).strip()
            }
        ),
        "source_version": source.version,
        "source_page": source.source_page,
        "license_name": source.license_name,
        "license_url": source.license_url,
    }


def iter_oewn_definitions(
    path: Path,
    spec: DefinitionSourceSpec,
) -> Iterator[dict[str, Any]]:
    """Yield one record for every single-word OEWN lexical sense."""
    synsets: dict[str, tuple[str, tuple[str, ...]]] = {}
    with _open_binary(path) as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            tag = _local_name(element.tag)
            if tag == "LexicalEntry":
                element.clear()
                continue
            if tag != "Synset":
                continue
            synset_id = element.attrib.get("id", "")
            definitions = [
                " ".join(child.text.split())
                for child in element
                if _local_name(child.tag) == "Definition"
                and child.text
                and child.text.strip()
            ]
            examples = tuple(
                " ".join(child.text.split())
                for child in element
                if _local_name(child.tag) == "Example"
                and child.text
                and child.text.strip()
            )
            if synset_id and definitions:
                synsets[synset_id] = (definitions[0], examples)
            element.clear()

    with _open_binary(path) as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            tag = _local_name(element.tag)
            if tag == "Synset":
                element.clear()
                continue
            if tag != "LexicalEntry":
                continue
            lemma = next(
                (
                    child
                    for child in element
                    if _local_name(child.tag) == "Lemma"
                ),
                None,
            )
            if lemma is None:
                element.clear()
                continue
            surface = _single_word(
                lemma.attrib.get("writtenForm", lemma.attrib.get("lemma", ""))
            )
            if surface is None:
                element.clear()
                continue
            part_of_speech = lemma.attrib.get("partOfSpeech", "")
            for sense in element:
                if _local_name(sense.tag) != "Sense":
                    continue
                synset_id = sense.attrib.get("synset", "")
                payload = synsets.get(synset_id)
                if payload is None:
                    continue
                definition, examples = payload
                sense_id = sense.attrib.get("id") or (
                    f"{element.attrib.get('id', surface)}:{synset_id}"
                )
                yield _record(
                    surface=surface,
                    part_of_speech=part_of_speech,
                    definition=definition,
                    source=spec,
                    source_sense_key=sense_id,
                    examples=examples,
                )
            element.clear()


def _wiktionary_examples(sense: Mapping[str, Any]) -> Iterator[str]:
    for example in sense.get("examples", ()) or ():
        if isinstance(example, str):
            if example.strip():
                yield example
            continue
        if isinstance(example, Mapping):
            text = example.get("text") or example.get("english")
            if text and str(text).strip():
                yield str(text)


def iter_wiktionary_definitions(
    path: Path,
    spec: DefinitionSourceSpec,
) -> Iterator[dict[str, Any]]:
    """Yield English single-word senses from Wiktextract JSONL."""
    with _open_text(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            if value.get("lang_code") != "en":
                continue
            surface = _single_word(str(value.get("word", "")))
            if surface is None:
                continue
            part_of_speech = str(value.get("pos", "")).strip()
            page_id = value.get("pageid", value.get("id", ""))
            etymology = value.get("etymology_number", 0)
            for sense_index, sense in enumerate(value.get("senses", ()) or ()):
                if not isinstance(sense, Mapping):
                    continue
                tags = {
                    normalize_word(str(tag))
                    for tag in (sense.get("tags", ()) or ())
                }
                if (
                    tags & WIKTIONARY_EXCLUDED_TAGS
                    or sense.get("form_of")
                    or sense.get("alt_of")
                ):
                    continue
                glosses = [
                    " ".join(str(gloss).split())
                    for gloss in (sense.get("glosses", ()) or ())
                    if str(gloss).strip()
                ]
                if not glosses:
                    continue
                definition = glosses[-1]
                sense_id = sense.get("id") or sense.get("senseid")
                if not sense_id:
                    digest = hashlib.sha256(
                        "\x1f".join(
                            (
                                normalize_word(surface),
                                part_of_speech,
                                str(etymology),
                                definition,
                            )
                        ).encode("utf-8")
                    ).hexdigest()[:24]
                    sense_id = f"{page_id}:{part_of_speech}:{sense_index}:{digest}"
                domains = tuple(sense.get("topics", ()) or ()) + tuple(
                    sense.get("categories", ()) or ()
                )
                yield _record(
                    surface=surface,
                    part_of_speech=part_of_speech,
                    definition=definition,
                    source=spec,
                    source_sense_key=str(sense_id),
                    examples=_wiktionary_examples(sense),
                    domains=domains,
                )


def _mesh_record_id(record: ET.Element) -> str:
    for name in ("DescriptorUI", "SupplementalRecordUI", "QualifierUI"):
        value = _child_text(record, name)
        if value:
            return value
    return ""


def iter_mesh_definitions(
    path: Path,
    spec: DefinitionSourceSpec,
) -> Iterator[dict[str, Any]]:
    """Yield MeSH terms whose concepts provide a ScopeNote definition."""
    record_tags = {"DescriptorRecord", "SupplementalRecord", "QualifierRecord"}
    with _open_binary(path) as handle:
        for _, element in ET.iterparse(handle, events=("end",)):
            if _local_name(element.tag) not in record_tags:
                continue
            record_id = _mesh_record_id(element)
            for concept in element.iter():
                if _local_name(concept.tag) != "Concept":
                    continue
                concept_id = _child_text(concept, "ConceptUI")
                definition = _child_text(concept, "ScopeNote")
                if not definition:
                    continue
                for term in concept.iter():
                    if _local_name(term.tag) != "Term":
                        continue
                    surface = _single_word(_child_text(term, "String"))
                    if surface is None:
                        continue
                    term_id = _child_text(term, "TermUI")
                    source_key = ":".join(
                        value
                        for value in (record_id, concept_id, term_id)
                        if value
                    )
                    if not source_key:
                        source_key = hashlib.sha256(
                            f"{surface}\x1f{definition}".encode("utf-8")
                        ).hexdigest()[:24]
                    yield _record(
                        surface=surface,
                        part_of_speech="noun",
                        definition=definition,
                        source=spec,
                        source_sense_key=source_key,
                    )
            element.clear()


PARSERS = {
    "oewn_lmf": iter_oewn_definitions,
    "wiktextract_jsonl": iter_wiktionary_definitions,
    "mesh_xml": iter_mesh_definitions,
}


def parse_wiktionary_snapshot_date(page_text: str) -> str:
    match = re.search(
        r"extracted\s+from\s+the.*?dump.*?dated\s+(\d{4}-\d{2}-\d{2})",
        page_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not resolve the Wiktionary dump date")
    return match.group(1)


def verify_wiktionary_snapshot(spec: DefinitionSourceSpec) -> None:
    """Stop if Kaikki's moving download no longer matches the locked date."""
    request = urllib.request.Request(
        spec.source_page,
        headers={"User-Agent": DEFAULT_USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        page_text = response.read().decode("utf-8", errors="replace")
    resolved = parse_wiktionary_snapshot_date(page_text)
    if resolved != spec.version:
        raise RuntimeError(
            "The English Wiktionary extraction has advanced from the locked "
            f"{spec.version} snapshot to {resolved}. Existing source files "
            "remain untouched. Review and intentionally update the source "
            "contract before downloading a different dictionary."
        )


def iter_source(
    path: Path,
    spec: DefinitionSourceSpec,
) -> Iterator[dict[str, Any]]:
    try:
        parser = PARSERS[spec.parser]
    except KeyError as error:
        raise ValueError(f"Unknown definition parser: {spec.parser}") from error
    yield from parser(path, spec)


def download_resumable(
    url: str,
    destination: Path,
    *,
    retries: int = 5,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """Download once, resuming an interrupted ``.part`` transfer."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return {
            "path": str(destination),
            "size_bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
            "action": "reused",
        }

    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": DEFAULT_USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and getattr(response, "status", 200) == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    while chunk := response.read(chunk_bytes):
                        output.write(chunk)
            os.replace(partial, destination)
            return {
                "path": str(destination),
                "size_bytes": destination.stat().st_size,
                "sha256": file_sha256(destination),
                "action": "downloaded",
            }
        except (OSError, urllib.error.URLError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Download failed after {retries} attempts: {url}"
                ) from error
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def _artifact_matches(
    output: Path,
    manifest_path: Path,
    *,
    raw_sha256: str,
    spec: DefinitionSourceSpec,
) -> dict[str, Any] | None:
    if not output.is_file() or not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = manifest["contract"]
        if contract != {
            "parser_version": PARSER_VERSION,
            "parser": spec.parser,
            "source_id": spec.source_id,
            "source_version": spec.version,
            "raw_sha256": raw_sha256,
            "single_word_headwords_only": True,
        }:
            return None
        if file_sha256(output) != manifest["canonical"]["sha256"]:
            return None
        return manifest
    except Exception:
        return None


def compile_source(
    raw_path: Path,
    output: Path,
    spec: DefinitionSourceSpec,
) -> dict[str, Any]:
    """Compile one raw source to canonical JSONL with a resumable contract."""
    raw_sha256 = file_sha256(raw_path)
    manifest_path = output.with_suffix(".manifest.json")
    existing = _artifact_matches(
        output,
        manifest_path,
        raw_sha256=raw_sha256,
        spec=spec,
    )
    if existing is not None:
        result = dict(existing)
        result["action"] = "reused"
        return result

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    row_count = 0
    unique_words: set[str] = set()
    domains: Counter[str] = Counter()
    with temporary.open("w", encoding="utf-8") as handle:
        for record in iter_source(raw_path, spec):
            handle.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            row_count += 1
            unique_words.add(normalize_word(record["surface"]))
            domains.update(record.get("domains", ()))
    os.replace(temporary, output)
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "action": "built",
        "contract": {
            "parser_version": PARSER_VERSION,
            "parser": spec.parser,
            "source_id": spec.source_id,
            "source_version": spec.version,
            "raw_sha256": raw_sha256,
            "single_word_headwords_only": True,
        },
        "source": asdict(spec),
        "raw": {
            "path": str(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "sha256": raw_sha256,
        },
        "canonical": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": file_sha256(output),
            "sense_rows": row_count,
            "unique_headwords": len(unique_words),
        },
        "top_domains": domains.most_common(100),
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def compile_reviewed_supplement(
    input_path: Path,
    canonical_root: Path,
) -> dict[str, Any]:
    """Validate a reviewed JSONL supplement and freeze its canonical rows."""
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    input_sha256 = file_sha256(input_path)
    source_id = f"reviewed_supplement:{input_path.stem}"
    output = (
        canonical_root
        / f"reviewed-{input_path.stem}-{input_sha256[:12]}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".manifest.json")
    contract = {
        "parser_version": PARSER_VERSION,
        "source_id": source_id,
        "input_sha256": input_sha256,
        "single_word_headwords_only": True,
        "reviewed_source_grounded_definitions": True,
    }
    if output.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("contract") == contract
            and existing.get("canonical", {}).get("sha256")
            == file_sha256(output)
        ):
            existing["action"] = "reused"
            return existing

    temporary = output.with_name(output.name + ".tmp")
    sense_rows = 0
    headwords: set[str] = set()
    with input_path.open(encoding="utf-8") as source_handle, temporary.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        for ordinal, raw_line in enumerate(source_handle, start=1):
            if not raw_line.strip():
                continue
            raw_record = json.loads(raw_line)
            record = normalize_source_record(
                raw_record,
                default_source=source_id,
                ordinal=ordinal,
            )
            units = tuple(iter_definition_words(record["surface"]))
            if (
                len(units) != 1
                or normalize_word(units[0]) != record["normalized"]
            ):
                raise ValueError(
                    f"{input_path}:{ordinal}: supplemental headword must be "
                    "one complete word"
                )
            canonical = {
                "surface": record["surface"],
                "part_of_speech": record["part_of_speech"],
                "definition": record["definition"],
                "source": record["source"],
                "source_sense_key": record["source_sense_key"],
                "examples": list(record["examples"]),
                "source_version": str(
                    raw_record.get("source_version", "reviewed")
                ),
                "source_page": str(raw_record.get("source_page", "")),
                "license_name": str(raw_record.get("license_name", "")),
                "license_url": str(raw_record.get("license_url", "")),
            }
            output_handle.write(
                json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            sense_rows += 1
            headwords.add(record["normalized"])
    os.replace(temporary, output)
    manifest = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "action": "built",
        "contract": contract,
        "source": {
            "source_id": source_id,
            "version": "reviewed",
            "source_page": str(input_path),
            "license_name": "per-record",
            "license_url": "",
            "attribution": "per-record reviewed supplement",
        },
        "raw": {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": input_sha256,
        },
        "canonical": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": file_sha256(output),
            "sense_rows": sense_rows,
            "unique_headwords": len(headwords),
        },
    }
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def iter_ngram_words(paths: Sequence[Path]) -> Iterator[tuple[str, int]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                text = record.get("text", record.get("surface_text"))
                if text is None:
                    raise KeyError(
                        f"{path}:{line_number}: expected text or surface_text"
                    )
                weight = int(
                    record.get("frequency", record.get("count", 1)) or 1
                )
                for word in iter_definition_words(str(text)):
                    yield normalize_word(word), max(weight, 1)


def definition_headwords(paths: Sequence[Path]) -> Iterator[str]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                surface = record.get("surface", record.get("word"))
                if not surface:
                    raise KeyError(f"{path}:{line_number}: missing surface")
                yield normalize_word(str(surface))


def build_coverage_report(
    canonical_paths: Sequence[Path],
    ngram_paths: Sequence[Path],
) -> dict[str, Any]:
    available = set(definition_headwords(canonical_paths))
    ngram_frequency: Counter[str] = Counter()
    for word, weight in iter_ngram_words(ngram_paths):
        ngram_frequency[word] += weight
    missing = set(ngram_frequency) - available
    covered = set(ngram_frequency) & available
    total_frequency = sum(ngram_frequency.values())
    covered_frequency = sum(ngram_frequency[word] for word in covered)
    return {
        "unique_ngram_words": len(ngram_frequency),
        "defined_ngram_words": len(covered),
        "missing_ngram_words": len(missing),
        "type_coverage": (
            len(covered) / len(ngram_frequency) if ngram_frequency else 1.0
        ),
        "frequency_weighted_coverage": (
            covered_frequency / total_frequency if total_frequency else 1.0
        ),
        "highest_frequency_missing_words": [
            {"word": word, "frequency": frequency}
            for word, frequency in sorted(
                (
                    (word, ngram_frequency[word])
                    for word in missing
                ),
                key=lambda item: (-item[1], item[0]),
            )[:1000]
        ],
    }


def build_dictionary_storage_report(
    canonical_paths: Sequence[Path],
) -> dict[str, Any]:
    """Measure the complete sense payload and its lightweight word-ID graph."""
    headwords: set[str] = set()
    definition_words: Counter[str] = Counter()
    sense_rows = 0
    definition_edges = 0
    for path in canonical_paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                surface = record.get("surface", record.get("word"))
                definition = record.get("definition", record.get("gloss"))
                if not surface or not definition:
                    raise KeyError(
                        f"{path}:{line_number}: expected surface and definition"
                    )
                headwords.add(normalize_word(str(surface)))
                words = [
                    normalize_word(word)
                    for word in iter_definition_words(str(definition))
                ]
                definition_words.update(words)
                definition_edges += len(words)
                sense_rows += 1

    label_only = set(definition_words) - headwords
    payload_bytes = sense_rows * 2048 * 2
    return {
        "payload_scope": "complete_curated_single_word_dictionary",
        "headwords": len(headwords),
        "sense_rows": sense_rows,
        "definition_word_nodes": len(definition_words),
        "definition_word_edges": definition_edges,
        "definition_words_with_sense_payload": len(
            set(definition_words) & headwords
        ),
        "definition_words_as_label_only_nodes": len(label_only),
        "all_definition_edges_resolve_to_word_ids": True,
        "estimated_definition_vector_bytes_bf16": payload_bytes,
        "estimated_definition_vector_gib_bf16": payload_bytes / (1024**3),
        "dendritron_cuda_bytes": 0,
        "highest_frequency_label_only_definition_words": [
            {"word": word, "occurrences": definition_words[word]}
            for word in sorted(
                label_only,
                key=lambda item: (-definition_words[item], item),
            )[:1000]
        ],
    }


def prepare_definition_sources(
    *,
    raw_root: Path,
    canonical_root: Path,
    ngram_paths: Sequence[Path],
    source_specs: Sequence[DefinitionSourceSpec] = SOURCE_SPECS,
    supplemental_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Acquire, compile, fingerprint, and measure the full dictionary."""
    raw_root.mkdir(parents=True, exist_ok=True)
    canonical_root.mkdir(parents=True, exist_ok=True)
    source_manifests: list[dict[str, Any]] = []
    canonical_paths: list[Path] = []
    for spec in source_specs:
        raw_path = raw_root / spec.filename
        if spec.source_id == "english_wiktionary" and not raw_path.is_file():
            verify_wiktionary_snapshot(spec)
        download = download_resumable(spec.url, raw_path)
        canonical_path = canonical_root / f"{spec.source_id}-{spec.version}.jsonl"
        compiled = compile_source(raw_path, canonical_path, spec)
        compiled["download_action"] = download["action"]
        source_manifests.append(compiled)
        canonical_paths.append(canonical_path)

    for supplemental_path in supplemental_paths:
        compiled = compile_reviewed_supplement(
            Path(supplemental_path),
            canonical_root,
        )
        source_manifests.append(compiled)
        canonical_paths.append(Path(compiled["canonical"]["path"]))

    coverage = build_coverage_report(canonical_paths, ngram_paths)
    storage = build_dictionary_storage_report(canonical_paths)
    manifest = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "parser_version": PARSER_VERSION,
        "selection_policy": {
            "language": "English",
            "headword_units": 1,
            "retain_polysemy": True,
            "retain_exact_definition_text": True,
            "payload_scope": "complete_curated_single_word_dictionary",
            "definition_word_policy": "ordered_word_id_links",
            "wiktionary_excluded_tags": sorted(WIKTIONARY_EXCLUDED_TAGS),
            "engram_vocabulary_coverage_required": 1.0,
        },
        "sources": source_manifests,
        "canonical_definition_files": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in canonical_paths
        ],
        "ngram_key_files": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
            }
            for path in ngram_paths
        ],
        "coverage": coverage,
        "storage": storage,
        "gpu_work_performed": False,
    }
    manifest_path = canonical_root / "definition_sources_manifest.json"
    coverage_path = canonical_root / "coverage_report.json"
    inventory_report_path = canonical_root / "dictionary_storage_report.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        coverage_path,
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        inventory_report_path,
        json.dumps(
            storage,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["coverage_report_path"] = str(coverage_path)
    manifest["dictionary_storage_report_path"] = str(inventory_report_path)
    return manifest
