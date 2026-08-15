"""Fit Dendritron source maps into frozen layer-2 definition geometry.

The fitter accepts either the historical NPZ archive or the sharded real-data
directory written by ``modal_extract_jtd_latent_assets.py``.

The historical NPZ contains same-content anchor pairs:

``layer8_source`` / ``layer8_reference``
``layer24_source`` / ``layer24_reference``
``live_source`` / ``live_reference``

The real-data directory contains ``anchors/bigrams.safetensors`` and
``anchors/trigrams.safetensors`` with row-aligned ``layer08``, ``layer24``, and
``layer02`` tensors. Every layer-2 row is a fixed Qwen reference location.
Surface words and IDs are absent from this numerical fitting stage.

The layer-8 and layer-24 maps can be fitted immediately after donor-anchor
extraction.  The live-state map is initialized as identity at production width
and remains trainable with the recipient unless explicit live/reference pairs
are supplied.
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


def _optional_pair(archive: Any, source: str) -> tuple[Tensor, Tensor] | None:
    source_key = f"{source}_source"
    reference_key = f"{source}_reference"
    present = (source_key in archive, reference_key in archive)
    if present == (False, False):
        return None
    if present != (True, True):
        raise KeyError(
            f"Anchor archive must contain both {source_key} and {reference_key}"
        )
    return _load_pair(archive, source)


def _load_real_anchor_root(
    root: Path,
) -> tuple[Tensor, Tensor, Tensor]:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - host dependency contract
        raise RuntimeError(
            "Real JTD anchors require safetensors>=0.5"
        ) from error

    layer8_parts = []
    layer24_parts = []
    reference_parts = []
    for bank_name in ("bigrams", "trigrams"):
        path = root / "anchors" / f"{bank_name}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {
                "row_indices",
                "layer02",
                "layer08",
                "layer24",
            }:
                raise ValueError(f"Unexpected JTD anchor tensors in {path}")
            row_indices = handle.get_tensor("row_indices")
            layer02 = handle.get_tensor("layer02")
            layer08 = handle.get_tensor("layer08")
            layer24 = handle.get_tensor("layer24")
            if row_indices.ndim != 1:
                raise ValueError(f"row_indices must be one-dimensional: {path}")
            if row_indices.shape[0] < 2:
                raise ValueError(f"At least two JTD anchors are required: {path}")
            if bool((row_indices < 0).any()) or bool(
                (row_indices[1:] <= row_indices[:-1]).any()
            ):
                raise ValueError(
                    f"row_indices must be nonnegative and strictly increasing: {path}"
                )
            if not (
                layer02.ndim == layer08.ndim == layer24.ndim == 2
                and layer02.shape == layer08.shape == layer24.shape
                and layer02.shape[0] == row_indices.shape[0]
            ):
                raise ValueError(f"Row-aligned JTD anchor shapes differ: {path}")
            layer8_parts.append(layer08.to(torch.float32))
            layer24_parts.append(layer24.to(torch.float32))
            reference_parts.append(layer02.to(torch.float32))
    return (
        torch.cat(layer8_parts, dim=0),
        torch.cat(layer24_parts, dim=0),
        torch.cat(reference_parts, dim=0),
    )


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
    if steps < 1:
        raise ValueError("steps must be positive")
    if batch_size < 2:
        raise ValueError("batch_size must be at least two")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if neighbor_count < 1:
        raise ValueError("neighbor_count must be positive")
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
    anchor_root = getattr(args, "anchor_root", None)
    anchor_archive = getattr(args, "anchors", None)
    live_pair: tuple[Tensor, Tensor] | None = None
    if anchor_root is not None:
        layer8, layer24, shared_reference = _load_real_anchor_root(anchor_root)
        reference8 = shared_reference
        reference24 = shared_reference
        anchor_description = str(anchor_root)
    else:
        if anchor_archive is None:
            raise ValueError("Either anchors or anchor_root is required")
        with np.load(anchor_archive, allow_pickle=False) as archive:
            layer8, reference8 = _load_pair(archive, "layer8")
            layer24, reference24 = _load_pair(archive, "layer24")
            live_pair = _optional_pair(archive, "live")
        anchor_description = str(anchor_archive)

    joint_widths = {
        reference8.shape[1],
        reference24.shape[1],
    }
    if live_pair is not None:
        joint_widths.add(live_pair[1].shape[1])
    if len(joint_widths) != 1:
        raise ValueError("All layer-2 reference widths must match")
    joint_width = joint_widths.pop()
    if layer8.shape[1] != joint_width or layer24.shape[1] != joint_width:
        raise ValueError("Layer-8 and layer-24 source widths must match layer 2")

    if live_pair is None:
        live_width = int(getattr(args, "live_width", None) or joint_width)
        if live_width != joint_width:
            raise ValueError(
                "A nonmatching live width requires explicit live/reference anchors"
            )
        live = None
        reference_live = None
    else:
        live, reference_live = live_pair
        live_width = int(live.shape[1])

    jtd = JointTransferDomain(
        live_width,
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
    }
    if live is None or reference_live is None:
        reports["live"] = {
            "status": "identity_initialized_train_with_recipient",
            "loss_initial": None,
            "loss_final": None,
        }
    else:
        reports["live"] = {
            "status": "fitted_from_explicit_pairs",
            **_fit_map(
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
        "schema_version": 2,
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
        "schema_version": 2,
        "completed": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "anchors": anchor_description,
        "checkpoint": str(args.output),
        "joint_width": joint_width,
        "live_width": live_width,
        "rows": {
            "layer8": layer8.shape[0],
            "layer24": layer24.shape[0],
            "live": 0 if live is None else live.shape[0],
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--anchors", type=Path)
    source.add_argument("--anchor-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--live-width",
        type=int,
        help=(
            "Recipient width when live anchors are absent. It must equal the "
            "2,048D reference width for identity initialization."
        ),
    )
    parser.add_argument("--steps-per-source", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--neighbor-count", type=int, default=8)
    return parser


def main() -> None:
    print(json.dumps(fit(build_parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
