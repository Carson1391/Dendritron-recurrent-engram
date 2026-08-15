"""Run the punctuation-aware Engram inventory comparison on the Modal volume.

Run after the corrected Stage-1 recount:

    modal run modal_compare_punctuation_inventory.py
"""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import modal

from modal_extract_states import CORPUS_VOLUME_NAME, DATA_MOUNT


APP_NAME = "dendritron-punctuation-inventory-audit"
OLD_STAGE2 = DATA_MOUNT / "dendritron-stage2"
NEW_STAGE1 = DATA_MOUNT / "dendritron-stage1-punctuation-v2/final"
OUTPUT = DATA_MOUNT / "dendritron-stage1-punctuation-v2/audit"

app = modal.App(APP_NAME)
corpus_volume = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12")


@app.function(
    image=image,
    cpu=4.0,
    memory=8_192,
    timeout=60 * 60,
    volumes={str(DATA_MOUNT): corpus_volume},
)
def compare_on_volume() -> dict:
    from stage3_jtd.compare_punctuation_inventories import compare

    result = compare(
        Namespace(
            old_stage2=OLD_STAGE2,
            new_stage1=NEW_STAGE1,
            output=OUTPUT,
        )
    )
    corpus_volume.commit()
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(compare_on_volume.remote(), indent=2, sort_keys=True))
