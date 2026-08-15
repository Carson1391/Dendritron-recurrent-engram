"""Optional CPU PCA diagnostic for dictionary layer-2 rows.

The dictionary bank already provides exact sparse sense lookup. Dendritron's
Universal/Shared-LoRA subspace comes from task-adapter updates in weight space,
so this definition-vector PCA is excluded from the required model build. It is
kept only as an optional diagnostic for studying the geometry of donor rows.

Example on the Modal volume:

    python stage4_subspace/build_universal_subspace.py \
        --bank /data/dendritron-stage3-dictionary/bank \
        --output /data/dendritron-diagnostics/definition-vector-pca \
        --rank auto \
        --variance-target 0.98 \
        --device cpu

The expensive full eigensystem is cached by dictionary-bank fingerprint.
Changing the retained rank reuses that cache and only rewrites the compact
basis and coordinate shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dendritron.universal_subspace import (  # noqa: E402
    DEFAULT_EXPLAINED_VARIANCE_TARGET,
    DEFAULT_KNOWLEDGE_RANK_CANDIDATES,
    select_knowledge_rank,
)


VECTOR_KEY = "layer02"
EXPECTED_WIDTH = 2048
DEFAULT_RANK = "auto"
DEFAULT_BATCH_ROWS = 4096
RANK_CANDIDATES = DEFAULT_KNOWLEDGE_RANK_CANDIDATES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def discover_shards(bank: Path) -> list[Path]:
    shards = sorted((bank / "shards").glob("shard-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No definition shards found under {bank / 'shards'}")
    return shards


def shard_shape(path: Path) -> tuple[int, int]:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {VECTOR_KEY}:
            raise ValueError(f"{path} must contain only {VECTOR_KEY}")
        shape = tuple(handle.get_slice(VECTOR_KEY).get_shape())
    if len(shape) != 2 or shape[1] != EXPECTED_WIDTH:
        raise ValueError(
            f"{path} has shape {shape}; expected [rows, {EXPECTED_WIDTH}]"
        )
    return int(shape[0]), int(shape[1])


def source_contract(bank: Path, shards: Sequence[Path]) -> dict[str, Any]:
    manifest_path = bank / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dictionary manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["hidden_state_layer"]) != 2:
        raise ValueError("Definition-vector PCA source must be dictionary layer 2")
    shard_rows = [shard_shape(path)[0] for path in shards]
    if sum(shard_rows) != int(manifest["rows"]):
        raise ValueError("Dictionary manifest row count differs from tensor shards")
    return {
        "bank_manifest_path": str(manifest_path),
        "bank_manifest_sha256": file_sha256(manifest_path),
        "rows": sum(shard_rows),
        "width": EXPECTED_WIDTH,
        "shards": [
            {
                "path": str(path),
                "rows": rows,
                "sha256": file_sha256(path),
            }
            for path, rows in zip(shards, shard_rows, strict=True)
        ],
    }


def iter_tensor_batches(
    shards: Sequence[Path],
    *,
    batch_rows: int,
    device: str,
) -> Iterator[Any]:
    import torch
    from safetensors import safe_open

    for path in shards:
        with safe_open(path, framework="pt", device="cpu") as handle:
            rows = int(handle.get_slice(VECTOR_KEY).get_shape()[0])
            for start in range(0, rows, batch_rows):
                stop = min(start + batch_rows, rows)
                values = handle.get_slice(VECTOR_KEY)[start:stop]
                yield values.to(device=device, dtype=torch.float32)


def fit_eigensystem(
    shards: Sequence[Path],
    *,
    batch_rows: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    vector_sum = torch.zeros(EXPECTED_WIDTH, device=device, dtype=torch.float64)
    second_sum = torch.zeros(
        EXPECTED_WIDTH,
        EXPECTED_WIDTH,
        device=device,
        dtype=torch.float64,
    )
    count = 0
    for values in iter_tensor_batches(
        shards,
        batch_rows=batch_rows,
        device=device,
    ):
        values = torch.nn.functional.normalize(values, dim=1)
        values64 = values.to(torch.float64)
        vector_sum.add_(values64.sum(dim=0))
        second_sum.add_(values64.T @ values64)
        count += values.shape[0]

    mean = vector_sum / count
    covariance = second_sum / count - torch.outer(mean, mean)
    covariance = (covariance + covariance.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = torch.clamp(eigenvalues[order], min=0)
    eigenvectors = eigenvectors[:, order]
    ratio = eigenvalues / eigenvalues.sum().clamp_min(1e-30)

    rank_report = {}
    cumulative = torch.cumsum(ratio, dim=0)
    for candidate in RANK_CANDIDATES:
        if candidate <= EXPECTED_WIDTH:
            rank_report[str(candidate)] = {
                "explained_variance": float(cumulative[candidate - 1].item()),
                "residual_variance": float(1.0 - cumulative[candidate - 1].item()),
            }

    tensors = {
        "mean": mean.to(device="cpu", dtype=torch.float32),
        "components": eigenvectors.to(device="cpu", dtype=torch.float32),
        "eigenvalues": eigenvalues.to(device="cpu", dtype=torch.float32),
        "explained_variance_ratio": ratio.to(device="cpu", dtype=torch.float32),
    }
    report = {
        "rows": count,
        "width": EXPECTED_WIDTH,
        "input_normalization": "row_l2",
        "centering": "global_mean_after_l2",
        "rank_report": rank_report,
    }
    return tensors, report


def write_eigensystem(
    output: Path,
    tensors: dict[str, Any],
) -> Path:
    from safetensors.torch import save_file

    path = output / "universal_knowledge_eigensystem.safetensors"
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(temporary))
    os.replace(temporary, path)
    return path


def load_eigensystem(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    expected = {
        "mean": (EXPECTED_WIDTH,),
        "components": (EXPECTED_WIDTH, EXPECTED_WIDTH),
        "eigenvalues": (EXPECTED_WIDTH,),
        "explained_variance_ratio": (EXPECTED_WIDTH,),
    }
    for name, shape in expected.items():
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise ValueError(f"Cached eigensystem tensor {name!r} has a bad shape")
    return tensors


def write_basis(
    output: Path,
    tensors: dict[str, Any],
) -> Path:
    from safetensors.torch import save_file

    path = output / "universal_knowledge_basis.safetensors"
    temporary = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(temporary))
    os.replace(temporary, path)
    return path


def write_coordinates(
    shards: Sequence[Path],
    *,
    output: Path,
    basis_tensors: dict[str, Any],
    batch_rows: int,
    device: str,
    force: bool,
) -> list[dict[str, Any]]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    coordinate_root = output / "coordinates"
    coordinate_root.mkdir(parents=True, exist_ok=True)
    mean = basis_tensors["mean"].to(device=device, dtype=torch.float32)
    components = basis_tensors["components"].to(device=device, dtype=torch.float32)
    artifacts = []
    for shard_index, source_path in enumerate(shards):
        target = coordinate_root / f"shard-{shard_index:05d}.safetensors"
        source_rows = shard_shape(source_path)[0]
        if target.is_file() and not force:
            try:
                with safe_open(target, framework="pt", device="cpu") as handle:
                    shape = tuple(handle.get_slice("coordinates").get_shape())
                if shape == (source_rows, components.shape[1]):
                    artifacts.append(
                        {
                            "path": str(target),
                            "rows": source_rows,
                            "sha256": file_sha256(target),
                        }
                    )
                    continue
            except Exception:
                pass

        coordinate_batches = []
        with safe_open(source_path, framework="pt", device="cpu") as handle:
            source = handle.get_slice(VECTOR_KEY)
            for start in range(0, source_rows, batch_rows):
                stop = min(start + batch_rows, source_rows)
                values = source[start:stop].to(device=device, dtype=torch.float32)
                values = torch.nn.functional.normalize(values, dim=1)
                coordinates = (values - mean) @ components
                coordinate_batches.append(
                    coordinates.to(device="cpu", dtype=torch.float16)
                )
        complete = torch.cat(coordinate_batches, dim=0)
        temporary = target.with_name(target.name + ".tmp")
        save_file({"coordinates": complete.contiguous()}, str(temporary))
        os.replace(temporary, target)
        artifacts.append(
            {
                "path": str(target),
                "rows": source_rows,
                "sha256": file_sha256(target),
            }
        )
    return artifacts


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device != "cpu":
        raise ValueError("Definition-vector PCA is a CPU-only diagnostic")
    shards = discover_shards(args.bank)
    contract = source_contract(args.bank, shards)
    source_fingerprint = json_fingerprint(
        {
            "contract": contract,
            "normalization": "row_l2",
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)

    eigensystem_path = args.output / "universal_knowledge_eigensystem.safetensors"
    eigensystem_manifest_path = args.output / "eigensystem_manifest.json"
    eigensystem = None
    eigensystem_report: dict[str, Any] | None = None
    if (
        eigensystem_manifest_path.is_file()
        and eigensystem_path.is_file()
        and not args.recompute_eigensystem
    ):
        cached_manifest = json.loads(
            eigensystem_manifest_path.read_text(encoding="utf-8")
        )
        if (
            cached_manifest.get("source_fingerprint") == source_fingerprint
            and cached_manifest.get("sha256") == file_sha256(eigensystem_path)
        ):
            eigensystem = load_eigensystem(eigensystem_path)
            eigensystem_report = cached_manifest["fit"]

    if eigensystem is None:
        eigensystem, eigensystem_report = fit_eigensystem(
            shards,
            batch_rows=args.batch_rows,
            device=args.device,
        )
        eigensystem_path = write_eigensystem(args.output, eigensystem)
        atomic_write_text(
            eigensystem_manifest_path,
            json.dumps(
                {
                    "schema_version": 1,
                    "created_at_utc": utc_now(),
                    "source_fingerprint": source_fingerprint,
                    "sha256": file_sha256(eigensystem_path),
                    "fit": eigensystem_report,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    ratio = eigensystem["explained_variance_ratio"].numpy()
    if args.rank == "auto":
        rank_selection = select_knowledge_rank(
            ratio,
            target=args.variance_target,
            candidates=RANK_CANDIDATES,
            minimum_rank=16,
        )
        selected_rank = rank_selection.rank
        selection_report = {
            "mode": "variance_target",
            "target": rank_selection.target,
            "selected_rank": rank_selection.rank,
            "explained_variance": rank_selection.explained_variance,
            "target_reached": rank_selection.target_reached,
            "candidates": list(rank_selection.candidates),
        }
    else:
        selected_rank = int(args.rank)
        if not 1 <= selected_rank <= EXPECTED_WIDTH:
            raise ValueError(f"rank must be between 1 and {EXPECTED_WIDTH}")
        cumulative = float(ratio[:selected_rank].sum())
        selection_report = {
            "mode": "fixed",
            "target": args.variance_target,
            "selected_rank": selected_rank,
            "explained_variance": cumulative,
            "target_reached": cumulative >= args.variance_target,
            "candidates": list(RANK_CANDIDATES),
        }

    fingerprint = json_fingerprint(
        {
            "source_fingerprint": source_fingerprint,
            "rank": selected_rank,
            "normalization": "row_l2",
        }
    )
    existing_manifest_path = args.output / "manifest.json"
    if existing_manifest_path.is_file() and not args.rebuild_coordinates:
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        artifact_paths = [
            Path(existing.get("basis", {}).get("path", "")),
            *[
                Path(item.get("path", ""))
                for item in existing.get("coordinate_shards", [])
            ],
        ]
        if (
            existing.get("fingerprint") == fingerprint
            and artifact_paths
            and all(path.is_file() for path in artifact_paths)
        ):
            return existing

    basis_tensors = {
        "mean": eigensystem["mean"],
        "components": eigensystem["components"][:, :selected_rank].contiguous(),
        "eigenvalues": eigensystem["eigenvalues"],
        "explained_variance_ratio": eigensystem["explained_variance_ratio"],
    }
    basis_path = write_basis(args.output, basis_tensors)
    coordinates = write_coordinates(
        shards,
        output=args.output,
        basis_tensors=basis_tensors,
        batch_rows=args.batch_rows,
        device=args.device,
        force=args.rebuild_coordinates,
    )
    manifest = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "fingerprint": fingerprint,
        "source_fingerprint": source_fingerprint,
        "role": "optional_definition_vector_pca_diagnostic",
        "used_by_dendritron_runtime": False,
        "dendritron_cuda_required": False,
        "definition_hidden_state_layer": 2,
        "source": contract,
        "fit": eigensystem_report,
        "rank_selection": selection_report,
        "eigensystem": {
            "path": str(eigensystem_path),
            "sha256": file_sha256(eigensystem_path),
        },
        "basis": {
            "path": str(basis_path),
            "sha256": file_sha256(basis_path),
        },
        "coordinate_shards": coordinates,
        "separate_skill_subspace": {
            "rank_range": [16, 32],
            "source": "successful_task_lora_factors",
        },
    }
    atomic_write_text(
        existing_manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rank",
        default=DEFAULT_RANK,
        help="Retained rank as an integer, or 'auto' to use the variance target",
    )
    parser.add_argument(
        "--variance-target",
        type=float,
        default=DEFAULT_EXPLAINED_VARIANCE_TARGET,
    )
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--recompute-eigensystem",
        action="store_true",
        help="Recompute covariance/eigendecomposition for the same source bank",
    )
    parser.add_argument(
        "--rebuild-coordinates",
        action="store_true",
        help="Rewrite coordinate shards while retaining the cached eigensystem",
    )
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
