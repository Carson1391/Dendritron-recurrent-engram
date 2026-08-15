"""Fit Dendritron's numerical JTD maps on CPU inside the Modal data volume.

This wrapper keeps Qwen out of the fitting job. It consumes the real anchor
Safetensors produced by ``modal_extract_jtd_latent_assets.py`` and writes a
small projection checkpoint plus JSON report.

Run:
    modal run modal_fit_joint_transfer_domain.py
"""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import modal

from modal_extract_states import CORPUS_VOLUME_NAME, DATA_MOUNT


APP_NAME = "dendritron-jtd-cpu-fit"
DEFAULT_ANCHOR_ROOT = "/data/dendritron-stage4-jtd-punctuation-v2/latent-assets"
DEFAULT_OUTPUT = "/data/dendritron-stage4-jtd-punctuation-v2/jtd-projections.pt"

app = modal.App(APP_NAME)
corpus_volume = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=2.1,<3",
        "safetensors>=0.5,<1",
        "torch>=2.7,<3",
    )
    .add_local_file(
        local_path=Path(__file__).parent / "modal_extract_states.py",
        remote_path="/root/modal_extract_states.py",
    )
    .add_local_dir(
        local_path=Path(__file__).parent / "stage3_jtd",
        remote_path="/root/stage3_jtd",
    )
    .add_local_dir(
        local_path=Path(__file__).parent / "dendritron",
        remote_path="/root/dendritron",
    )
)


def _validated_data_path(value: str, *, suffix: str | None = None) -> Path:
    path = Path(value)
    if not path.is_absolute() or DATA_MOUNT not in path.parents:
        raise ValueError("JTD paths must be located under /data")
    if not any(parent.name.startswith("dendritron-stage4-jtd") for parent in path.parents):
        raise ValueError("JTD paths must remain inside a dendritron-stage4-jtd root")
    if suffix is not None and path.suffix != suffix:
        raise ValueError(f"JTD output must end in {suffix}")
    return path


@app.function(
    image=image,
    cpu=16.0,
    memory=65_536,
    timeout=24 * 60 * 60,
    volumes={str(DATA_MOUNT): corpus_volume},
)
def fit_on_volume(
    anchor_root: str = str(DEFAULT_ANCHOR_ROOT),
    output: str = str(DEFAULT_OUTPUT),
    steps_per_source: int = 2_000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    neighbor_count: int = 8,
) -> dict:
    from stage3_jtd.fit_joint_transfer_domain import fit

    resolved_anchors = _validated_data_path(anchor_root)
    resolved_output = _validated_data_path(output, suffix=".pt")
    if not resolved_anchors.is_dir():
        raise FileNotFoundError(resolved_anchors)
    result = fit(
        Namespace(
            anchors=None,
            anchor_root=resolved_anchors,
            output=resolved_output,
            live_width=2_048,
            steps_per_source=int(steps_per_source),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            neighbor_count=int(neighbor_count),
        )
    )
    corpus_volume.commit()
    return result


@app.local_entrypoint()
def main(
    anchor_root: str = str(DEFAULT_ANCHOR_ROOT),
    output: str = str(DEFAULT_OUTPUT),
    steps_per_source: int = 2_000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    neighbor_count: int = 8,
) -> None:
    print(
        json.dumps(
            fit_on_volume.remote(
                anchor_root=anchor_root,
                output=output,
                steps_per_source=steps_per_source,
                batch_size=batch_size,
                learning_rate=learning_rate,
                neighbor_count=neighbor_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )
