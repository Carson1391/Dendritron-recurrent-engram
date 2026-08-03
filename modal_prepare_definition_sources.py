"""Build the real Dendritron definition inventory inputs on a Modal CPU.

Run:

    modal run modal_prepare_definition_sources.py

The job downloads versioned lexical sources, compiles the complete curated
single-word dictionary with every retained sense, and measures mandatory
coverage against the completed Stage 2 Engram vocabulary. It loads no model
and requests CPU resources only.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from modal_extract_states import CORPUS_VOLUME_NAME, DATA_MOUNT


APP_NAME = "dendritron-definition-sources"
SOURCE_ROOT = DATA_MOUNT / "dendritron-stage3-definition-sources"
RAW_ROOT = SOURCE_ROOT / "raw"
CANONICAL_ROOT = SOURCE_ROOT / "canonical"
BIGRAM_KEYS = DATA_MOUNT / "dendritron-stage2/bigrams/keys.jsonl"
TRIGRAM_KEYS = DATA_MOUNT / "dendritron-stage2/trigrams/keys.jsonl"

app = modal.App(APP_NAME)
corpus_volume = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(
    image=image,
    cpu=8.0,
    memory=65_536,
    timeout=24 * 60 * 60,
    volumes={str(DATA_MOUNT): corpus_volume},
)
def build_sources(supplemental_definitions: tuple[str, ...] = ()) -> dict:
    from stage3_dictionary.definition_sources import prepare_definition_sources

    for path in (BIGRAM_KEYS, TRIGRAM_KEYS):
        if not path.is_file():
            raise FileNotFoundError(f"Completed Stage 2 key file is required: {path}")
    result = prepare_definition_sources(
        raw_root=RAW_ROOT,
        canonical_root=CANONICAL_ROOT,
        ngram_paths=(BIGRAM_KEYS, TRIGRAM_KEYS),
        supplemental_paths=tuple(
            Path(path) for path in supplemental_definitions
        ),
    )
    corpus_volume.commit()
    return {
        "manifest_path": result["manifest_path"],
        "coverage_report_path": result["coverage_report_path"],
        "dictionary_storage_report_path": result[
            "dictionary_storage_report_path"
        ],
        "source_rows": {
            source["source"]["source_id"]: source["canonical"]["sense_rows"]
            for source in result["sources"]
        },
        "coverage": result["coverage"],
        "storage": result["storage"],
        "gpu_work_performed": False,
    }


@app.local_entrypoint()
def main(supplemental_definitions: str = "") -> None:
    supplements = tuple(
        value.strip()
        for value in supplemental_definitions.split(",")
        if value.strip()
    )
    print(
        json.dumps(
            build_sources.remote(supplements),
            ensure_ascii=False,
            indent=2,
        )
    )
