#!/usr/bin/env python3
r"""
qwen_lora_universal_directions.py

Paper-style Universal Subspace direction finder for Qwen LoRA adapters.

This is NOT single-model row PCA.

It does the paper-shaped operation:

  same architecture / same layer / same module / many tasks-adapters
  -> collect LoRA delta matrices ΔW = (alpha/r) * B @ A
  -> stack into tensor X with shape [num_adapters, out_dim, in_dim]
  -> zero-center
  -> mode-wise SVD / HOSVD-style factors
  -> save principal directions and explained-variance rank sweep

For Q/K/V hidden-state work, the important factor is:

  U_in[layer,module]  ∈ R^(in_dim x rank)

because q_proj/k_proj/v_proj weights are shaped:

  out_dim x in_dim

and U_in gives directions in the input hidden space.

Outputs:
  run_metadata.json
  adapter_index.csv
  rank_sweep.csv
  module_summary.csv
  directions/*.pt

Example, WSL:
  python qwen_lora_universal_directions.py \
    --adapters_dir /home/carson1391/qwen_loras \
    --out_dir /home/carson1391/RockBottom/qwen_universal_lora_dirs \
    --modules q_proj,k_proj,v_proj \
    --layers all \
    --ranks 8,16,32,64,128,256,512 \
    --center task \
    --matrix delta

Example, PowerShell:
  python qwen_lora_universal_directions.py `
    --adapters_dir "C:\Users\carso\Desktop\qwen_loras" `
    --out_dir "C:\Users\carso\Desktop\RockBottom\qwen_universal_lora_dirs" `
    --modules q_proj,k_proj,v_proj `
    --layers all `
    --ranks 8,16,32,64,128,256,512 `
    --center task `
    --matrix delta
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
MODULE_RE_TEMPLATE = r"\.({module})\."
A_RE = re.compile(r"lora_A(?:\.default)?\.weight$")
B_RE = re.compile(r"lora_B(?:\.default)?\.weight$")


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def find_adapter_files(adapters_dir: Path) -> List[Path]:
    names = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
    ]
    files: List[Path] = []
    for name in names:
        files.extend(adapters_dir.rglob(name))
    return sorted(set(files))


def load_tensor_file(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("safetensors is required for .safetensors adapters. Install safetensors.") from exc
        return load_file(str(path), device="cpu")
    blob = torch.load(str(path), map_location="cpu")
    if isinstance(blob, dict) and "state_dict" in blob and isinstance(blob["state_dict"], dict):
        return blob["state_dict"]
    if isinstance(blob, dict):
        return blob
    raise RuntimeError(f"Unsupported tensor file structure: {path}")


def adapter_config_for_file(adapter_file: Path) -> Dict[str, Any]:
    cfg_path = adapter_file.parent / "adapter_config.json"
    if cfg_path.exists():
        return read_json(cfg_path)
    return {}


def parse_layer(key: str) -> Optional[int]:
    m = LAYER_RE.search(key)
    return int(m.group(1)) if m else None


def key_has_module(key: str, module: str) -> bool:
    return re.search(MODULE_RE_TEMPLATE.format(module=re.escape(module)), key) is not None


def is_a_key(key: str) -> bool:
    return A_RE.search(key) is not None


def is_b_key(key: str) -> bool:
    return B_RE.search(key) is not None


def normalize_pair_base(key: str) -> str:
    key = re.sub(r"\.lora_A(?:\.default)?\.weight$", "", key)
    key = re.sub(r"\.lora_B(?:\.default)?\.weight$", "", key)
    return key


def get_alpha_scale(config: Dict[str, Any], module: str, rank: int) -> float:
    # PEFT usually stores global lora_alpha and r.
    alpha = config.get("lora_alpha", None)
    r_cfg = config.get("r", None)

    # Some configs may store rank/alpha patterns.
    alpha_pattern = config.get("alpha_pattern", {}) or {}
    rank_pattern = config.get("rank_pattern", {}) or {}

    if isinstance(alpha_pattern, dict):
        for pat, val in alpha_pattern.items():
            if module in pat or pat in module:
                alpha = val
                break
    if isinstance(rank_pattern, dict):
        for pat, val in rank_pattern.items():
            if module in pat or pat in module:
                r_cfg = val
                break

    if alpha is None:
        alpha = rank
    if r_cfg is None:
        r_cfg = rank

    try:
        return float(alpha) / float(r_cfg)
    except Exception:
        return 1.0


def parse_layers_arg(text: str, available: Iterable[int]) -> List[int]:
    available_set = set(int(x) for x in available)
    if text.strip().lower() == "all":
        return sorted(available_set)

    out = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(x for x in out if x in available_set)


def center_stack(X: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X shape [N, O, I].
    task: subtract mean over adapter/task axis, shape [1,O,I].
    global: subtract scalar global mean.
    none: no centering.
    """
    if mode == "task":
        mu = X.mean(dim=0, keepdim=True)
        return X - mu, mu
    if mode == "global":
        mu = X.mean().reshape(1, 1, 1)
        return X - mu, mu
    if mode == "none":
        mu = torch.zeros((1, 1, 1), dtype=X.dtype)
        return X, mu
    raise ValueError(f"Unknown center mode: {mode}")


def svd_factor(M: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # M shape [features, samples]. Return left singular vectors and singular values.
    M = M.float()
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    return U.contiguous(), S.contiguous()


def explained_from_singular(S: torch.Tensor) -> torch.Tensor:
    power = S.float() ** 2
    total = power.sum()
    if total <= 0:
        return torch.zeros_like(power)
    return power / total


def rank_for_threshold(cum: torch.Tensor, threshold: float) -> int:
    idx = torch.nonzero(cum >= threshold, as_tuple=False)
    if idx.numel() == 0:
        return int(cum.numel())
    return int(idx[0].item() + 1)


def safe_rank(r: int, max_rank: int) -> int:
    return max(1, min(int(r), int(max_rank)))


def collect_lora_deltas(
    adapter_files: List[Path],
    modules: List[str],
    matrix_mode: str,
) -> Tuple[Dict[Tuple[int, str], List[Tuple[str, torch.Tensor]]], List[Dict[str, Any]]]:
    collected: Dict[Tuple[int, str], List[Tuple[str, torch.Tensor]]] = {}
    adapter_rows: List[Dict[str, Any]] = []

    for idx, f in enumerate(adapter_files):
        cfg = adapter_config_for_file(f)
        state = load_tensor_file(f)
        adapter_name = f.parent.name

        bases: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, tensor in state.items():
            if not torch.is_tensor(tensor):
                continue
            if not (is_a_key(key) or is_b_key(key)):
                continue
            base = normalize_pair_base(key)
            bases.setdefault(base, {})
            if is_a_key(key):
                bases[base]["A"] = tensor.detach().cpu().float()
                bases[base]["A_key"] = key  # type: ignore
            elif is_b_key(key):
                bases[base]["B"] = tensor.detach().cpu().float()
                bases[base]["B_key"] = key  # type: ignore

        found_count = 0
        for base, pair in bases.items():
            if "A" not in pair or "B" not in pair:
                continue
            layer = parse_layer(base)
            if layer is None:
                continue

            module_hit = None
            for module in modules:
                if key_has_module(base, module):
                    module_hit = module
                    break
            if module_hit is None:
                continue

            A = pair["A"]
            B = pair["B"]
            if A.dim() != 2 or B.dim() != 2:
                continue

            rank = int(A.shape[0])
            scale = get_alpha_scale(cfg, module_hit, rank)

            if matrix_mode == "delta":
                # A [r,in], B [out,r] -> ΔW [out,in]
                W = (B @ A) * scale
            elif matrix_mode == "A":
                W = A
            elif matrix_mode == "B":
                W = B
            else:
                raise ValueError(f"Unknown matrix mode: {matrix_mode}")

            collected.setdefault((layer, module_hit), []).append((adapter_name, W.contiguous()))
            found_count += 1

        adapter_rows.append({
            "adapter_index": idx,
            "adapter_name": adapter_name,
            "adapter_file": str(f),
            "found_lora_pairs": found_count,
            "matrix_mode": matrix_mode,
            "config_r": cfg.get("r"),
            "config_lora_alpha": cfg.get("lora_alpha"),
            "target_modules": json.dumps(cfg.get("target_modules"), ensure_ascii=False),
        })

    return collected, adapter_rows


def analyze_group(
    *,
    X_list: List[Tuple[str, torch.Tensor]],
    layer: int,
    module: str,
    center: str,
    ranks: List[int],
    thresholds: List[float],
    directions_dir: Path,
    save_max_rank: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    names = [n for n, _ in X_list]
    mats = [w for _, w in X_list]

    shapes = sorted(set(tuple(w.shape) for w in mats))
    if len(shapes) != 1:
        return [], {
            "layer": layer,
            "module": module,
            "status": "skipped_mixed_shapes",
            "num_adapters": len(mats),
            "shapes": json.dumps(shapes),
        }

    X = torch.stack(mats, dim=0).float()  # [N,O,I]
    Xc, mu = center_stack(X, center)
    N, O, I = Xc.shape

    # HOSVD-style mode unfoldings:
    # input mode directions: [I, N*O]
    M_in = Xc.permute(2, 0, 1).reshape(I, N * O)
    # output mode directions: [O, N*I]
    M_out = Xc.permute(1, 0, 2).reshape(O, N * I)
    # task/adaptor mode directions: [N, O*I]
    M_task = Xc.reshape(N, O * I)

    U_in, S_in = svd_factor(M_in)
    U_out, S_out = svd_factor(M_out)
    U_task, S_task = svd_factor(M_task)

    exp_in = explained_from_singular(S_in)
    exp_out = explained_from_singular(S_out)
    exp_task = explained_from_singular(S_task)

    cum_in = torch.cumsum(exp_in, dim=0)
    cum_out = torch.cumsum(exp_out, dim=0)
    cum_task = torch.cumsum(exp_task, dim=0)

    max_save_in = safe_rank(save_max_rank, U_in.shape[1])
    max_save_out = safe_rank(save_max_rank, U_out.shape[1])
    max_save_task = safe_rank(save_max_rank, U_task.shape[1])

    save_path = directions_dir / f"layer{layer:02d}_{module}_{center}.pt"
    torch.save({
        "layer": layer,
        "module": module,
        "center": center,
        "adapter_names": names,
        "shape": {"num_adapters": N, "out_dim": O, "in_dim": I},
        "mu": mu.cpu(),
        "U_in": U_in[:, :max_save_in].cpu(),
        "S_in": S_in.cpu(),
        "explained_in": exp_in.cpu(),
        "U_out": U_out[:, :max_save_out].cpu(),
        "S_out": S_out.cpu(),
        "explained_out": exp_out.cpu(),
        "U_task": U_task[:, :max_save_task].cpu(),
        "S_task": S_task.cpu(),
        "explained_task": exp_task.cpu(),
    }, save_path)

    rows: List[Dict[str, Any]] = []
    for mode_name, cum, exp, max_dim in [
        ("input_hidden", cum_in, exp_in, I),
        ("output", cum_out, exp_out, O),
        ("adapter_task", cum_task, exp_task, N),
    ]:
        for r in ranks:
            rr = safe_rank(r, cum.numel())
            rows.append({
                "layer": layer,
                "module": module,
                "mode": mode_name,
                "rank": rr,
                "requested_rank": r,
                "explained_variance": float(cum[rr - 1].item()) if rr > 0 else 0.0,
                "residual_variance": float(1.0 - cum[rr - 1].item()) if rr > 0 else 1.0,
                "top1_explained": float(exp[0].item()) if exp.numel() else 0.0,
                "num_adapters": N,
                "out_dim": O,
                "in_dim": I,
                "saved_file": str(save_path),
            })

    summary: Dict[str, Any] = {
        "layer": layer,
        "module": module,
        "status": "ok",
        "num_adapters": N,
        "out_dim": O,
        "in_dim": I,
        "saved_file": str(save_path),
        "input_hidden_top1": float(exp_in[0].item()) if exp_in.numel() else 0.0,
        "output_top1": float(exp_out[0].item()) if exp_out.numel() else 0.0,
        "adapter_task_top1": float(exp_task[0].item()) if exp_task.numel() else 0.0,
    }

    for th in thresholds:
        summary[f"input_hidden_rank_for_{int(th*100)}"] = rank_for_threshold(cum_in, th)
        summary[f"output_rank_for_{int(th*100)}"] = rank_for_threshold(cum_out, th)
        summary[f"adapter_task_rank_for_{int(th*100)}"] = rank_for_threshold(cum_task, th)

    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper-style Universal Subspace direction finder from Qwen LoRA adapters.")
    ap.add_argument("--adapters_dir", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--modules", default="q_proj,k_proj,v_proj", type=str)
    ap.add_argument("--layers", default="all", type=str)
    ap.add_argument("--ranks", default="8,16,32,64,128,256,512", type=str)
    ap.add_argument("--thresholds", default="0.90,0.95,0.98,0.99", type=str)
    ap.add_argument("--center", default="task", choices=["task", "global", "none"])
    ap.add_argument("--matrix", default="delta", choices=["delta", "A", "B"])
    ap.add_argument("--min_adapters", default=2, type=int)
    ap.add_argument("--save_max_rank", default=512, type=int)
    args = ap.parse_args()

    adapters_dir = Path(args.adapters_dir)
    out_dir = Path(args.out_dir) / f"universal_dirs_{stamp()}"
    directions_dir = out_dir / "directions"
    out_dir.mkdir(parents=True, exist_ok=True)
    directions_dir.mkdir(parents=True, exist_ok=True)

    modules = [x.strip() for x in args.modules.split(",") if x.strip()]
    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    adapter_files = find_adapter_files(adapters_dir)
    if not adapter_files:
        raise FileNotFoundError(f"No adapter_model.safetensors/bin files found under {adapters_dir}")

    print("=== Qwen LoRA Universal Direction Finder ===")
    print(f"Adapters dir: {adapters_dir}")
    print(f"Found adapters: {len(adapter_files)}")
    print(f"Modules: {modules}")
    print(f"Matrix mode: {args.matrix}")
    print(f"Center: {args.center}")
    print(f"Output: {out_dir.resolve()}")

    collected, adapter_rows = collect_lora_deltas(adapter_files, modules, args.matrix)
    write_csv(out_dir / "adapter_index.csv", adapter_rows)

    available_layers = sorted(set(k[0] for k in collected))
    selected_layers = parse_layers_arg(args.layers, available_layers)

    rank_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for layer in selected_layers:
        for module in modules:
            key = (layer, module)
            X_list = collected.get(key, [])
            if len(X_list) < int(args.min_adapters):
                summary_rows.append({
                    "layer": layer,
                    "module": module,
                    "status": "skipped_too_few_adapters",
                    "num_adapters": len(X_list),
                })
                continue

            print(f"Analyzing layer {layer} {module}: {len(X_list)} adapters")
            rows, summary = analyze_group(
                X_list=X_list,
                layer=layer,
                module=module,
                center=args.center,
                ranks=ranks,
                thresholds=thresholds,
                directions_dir=directions_dir,
                save_max_rank=int(args.save_max_rank),
            )
            rank_rows.extend(rows)
            summary_rows.append(summary)

    write_csv(out_dir / "rank_sweep.csv", rank_rows)
    write_csv(out_dir / "module_summary.csv", summary_rows)
    write_json(out_dir / "run_metadata.json", {
        "script": "qwen_lora_universal_directions.py",
        "purpose": "Paper-style universal subspace extraction from many LoRA adapters.",
        "adapters_dir": str(adapters_dir),
        "num_adapter_files": len(adapter_files),
        "modules": modules,
        "layers_arg": args.layers,
        "selected_layers": selected_layers,
        "ranks": ranks,
        "thresholds": thresholds,
        "center": args.center,
        "matrix": args.matrix,
        "min_adapters": int(args.min_adapters),
        "save_max_rank": int(args.save_max_rank),
        "outputs": {
            "adapter_index": str(out_dir / "adapter_index.csv"),
            "rank_sweep": str(out_dir / "rank_sweep.csv"),
            "module_summary": str(out_dir / "module_summary.csv"),
            "directions_dir": str(directions_dir),
        },
        "notes": [
            "This uses an adapter/task axis. It is not single-checkpoint PCA.",
            "For hidden-state projection, use U_in from the saved .pt files.",
            "For LoRA delta weights, ΔW = (alpha/r) * B @ A.",
            "Center=task subtracts the mean adapter delta per weight entry before SVD.",
        ],
    })

    print("\n=== Saved ===")
    print(out_dir.resolve())
    print("Key files:")
    print(f"  {out_dir / 'module_summary.csv'}")
    print(f"  {out_dir / 'rank_sweep.csv'}")
    print(f"  {directions_dir}")


if __name__ == "__main__":
    main()
