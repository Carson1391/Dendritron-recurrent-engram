"""Fit Dendritron source maps into frozen layer-2 definition geometry.

The input NPZ contains same-content anchor pairs:

``layer8_source`` / ``layer8_reference``
``layer24_source`` / ``layer24_reference``
``live_source`` / ``live_reference``

Every ``*_reference`` row is a Qwen layer-2 location.  References remain
fixed; the three source maps move into that frame.  Surface words and IDs are
absent from this numerical fitting stage.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except ImportError as error:  # pragma: no cover - execution host contract
    raise RuntimeError("Install CPU PyTorch 2.7+ to fit JTD maps") from error

from dendritron.joint_transfer import (
    JointTransferDomain,
    locality_preserving_alignment_loss,
)


def _load_pair(archive: Any, source: str) -> tuple[Tensor, Tensor]:
    source_key = f"{source}_source"
    reference_key = f"{source}_reference"
    if source_key not in archive or reference_key not in archive:
        raise KeyError(f"Anchor archive requires {source_key} and {reference_key}")
    values = torch.from_numpy(np.asarray(archive[source_key], dtype=np.float32))
    reference = torch.from_numpy(
        np.asarray(archive[reference_key], dtype=np.float32)
    )
    if values.ndim != 2 or reference.ndim != 2:
        raise ValueError(f"{source} anchors must be two-dimensional")
    if values.shape[0] != reference.shape[0]:
        raise ValueError(f"{source} source/reference row counts differ")
    if values.shape[0] < 2:
        raise ValueError(f"{source} requires at least two anchor rows")
    return values, reference


def _fit_map(
    mapping: nn.Linear,
    source: Tensor,
    reference: Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    neighbor_count: int,
    inverse_mapping: nn.Linear | None = None,
) -> dict[str, float]:
    parameters = list(mapping.parameters())
    if inverse_mapping is not None:
        parameters.extend(inverse_mapping.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    generator = torch.Generator().manual_seed(20260801)
    initial = None
    final = None
    for _ in range(int(steps)):
        indices = torch.randint(
            0,
            source.shape[0],
            (min(int(batch_size), source.shape[0]),),
            generator=generator,
        )
        batch_source = source.index_select(0, indices)
        batch_reference = reference.index_select(0, indices)
        optimizer.zero_grad(set_to_none=True)
        projected = mapping(batch_source)
        loss, _ = locality_preserving_alignment_loss(
            projected,
            batch_reference,
            neighbor_count=min(neighbor_count, len(indices) - 1),
        )
        if inverse_mapping is not None:
            loss = loss + F.mse_loss(
                inverse_mapping(batch_reference),
                batch_source,
            )
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        initial = value if initial is None else initial
        final = value
    return {
        "loss_initial": float(initial),
        "loss_final": float(final),
    }


def fit(args: argparse.Namespace) -> dict[str, Any]:
    with np.load(args.anchors, allow_pickle=False) as archive:
        layer8, reference8 = _load_pair(archive, "layer8")
        layer24, reference24 = _load_pair(archive, "layer24")
        live, reference_live = _load_pair(archive, "live")

    joint_widths = {
        reference8.shape[1],
        reference24.shape[1],
        reference_live.shape[1],
    }
    if len(joint_widths) != 1:
        raise ValueError("All layer-2 reference widths must match")
    joint_width = joint_widths.pop()
    if layer8.shape[1] != joint_width or layer24.shape[1] != joint_width:
        raise ValueError("Layer-8 and layer-24 source widths must match layer 2")

    jtd = JointTransferDomain(
        live.shape[1],
        memory_width=joint_width,
    ).cpu()
    reports = {
        "layer8": _fit_map(
            jtd.layer8_to_joint,
            layer8,
            reference8,
            steps=args.steps_per_source,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            neighbor_count=args.neighbor_count,
        ),
        "layer24": _fit_map(
            jtd.layer24_to_joint,
            layer24,
            reference24,
            steps=args.steps_per_source,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            neighbor_count=args.neighbor_count,
        ),
        "live": _fit_map(
            jtd.live_to_joint,
            live,
            reference_live,
            steps=args.steps_per_source,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            neighbor_count=args.neighbor_count,
            inverse_mapping=jtd.joint_to_live,
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "reference_frame": "frozen_qwen_layer2_definition_geometry",
        "layer8_to_joint": jtd.layer8_to_joint.weight.detach().cpu(),
        "layer24_to_joint": jtd.layer24_to_joint.weight.detach().cpu(),
        "live_to_joint": jtd.live_to_joint.weight.detach().cpu(),
        "joint_to_live": jtd.joint_to_live.weight.detach().cpu(),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    manifest = {
        "schema_version": 1,
        "completed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "anchors": str(args.anchors),
        "checkpoint": str(args.output),
        "joint_width": joint_width,
        "live_width": live.shape[1],
        "rows": {
            "layer8": layer8.shape[0],
            "layer24": layer24.shape[0],
            "live": live.shape[0],
        },
        "reports": reports,
        "definition_transform": "identity",
        "surface_metadata_in_vectors": False,
    }
    manifest_path = args.output.with_suffix(".json")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps-per-source", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--neighbor-count", type=int, default=8)
    return parser


def main() -> None:
    print(json.dumps(fit(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
