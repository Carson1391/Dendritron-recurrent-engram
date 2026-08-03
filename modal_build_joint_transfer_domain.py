"""CPU-only Modal launcher for the recipient surface-memory compiler.

Run:
    modal run modal_build_joint_transfer_domain.py

The command reads the completed donor banks and finalized dictionary inventory
from the existing data volume.  It tokenizes text keys and writes only CPU
index artifacts under ``/data/dendritron-stage4-jtd``. The historical command
name remains stable; latent JTD fitting is a separate CPU step.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import modal

from modal_extract_states import (
    CORPUS_VOLUME_NAME,
    DATA_MOUNT,
    MODEL_CACHE,
    MODEL_MOUNT,
    MODEL_VOLUME_NAME,
)


APP_NAME = "dendritron-jtd-compiler"
OUTPUT_ROOT = DATA_MOUNT / "dendritron-stage4-jtd"

app = modal.App(APP_NAME)
corpus_volume = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=False)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "huggingface-hub[hf-xet]>=0.34,<2",
        "sentencepiece>=0.2,<1",
        "tokenizers>=0.21,<1",
        "transformers>=5.4,<6",
    )
    .env(
        {
            "HF_HOME": "/models/huggingface",
            "HF_HUB_CACHE": "/models/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "true",
        }
    )
)


@app.function(
    image=image,
    cpu=8.0,
    memory=32_768,
    timeout=12 * 60 * 60,
    volumes={
        str(DATA_MOUNT): corpus_volume,
        str(MODEL_MOUNT): model_volume,
    },
)
def compile_on_volume(
    rebuild: bool = False,
) -> dict:
    from stage3_jtd.build_joint_transfer_domain import build

    result = build(
        Namespace(
            stage2_root=DATA_MOUNT / "dendritron-stage2",
            dictionary=(
                DATA_MOUNT
                / "dendritron-stage3-dictionary/inventory/dictionary.sqlite3"
            ),
            output=OUTPUT_ROOT,
            cache_dir=MODEL_CACHE,
            rebuild=bool(rebuild),
        )
    )
    corpus_volume.commit()
    model_volume.commit()
    return {
        "completed": result["completed"],
        "resume_action": result.get("resume_action", "built"),
        "source_fingerprint": result["source"]["fingerprint"],
        "tokenizer_fingerprint": result["tokenizer_contract"]["fingerprint"],
        "projection_fingerprint": result["canonical_token_projection"][
            "fingerprint"
        ],
        "qwen_projection_reduction_percent": result[
            "canonical_token_projection"
        ]["reduction_percent"],
        "rows": result["surface_index"]["source_rows"],
        "address_rows": result["surface_index"]["address_rows"],
        "maximum_token_span": result["surface_index"]["maximum_token_span"],
        "cryptographic_hash_collisions": result["surface_index"]["collision_report"][
            "cryptographic_hash_collisions"
        ],
    }


@app.local_entrypoint()
def main(
    rebuild: bool = False,
) -> None:
    print(
        json.dumps(
            compile_on_volume.remote(rebuild),
            indent=2,
        )
    )
