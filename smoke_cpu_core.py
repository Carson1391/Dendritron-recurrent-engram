"""CPU-only end-to-end learning smoke for the Dendritron reference core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from dendritron.config import tiny_smoke_config
from dendritron.model import DendritronLM
from dendritron.output_geometry import rank_margin_loss


def windows(values: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = []
    targets = []
    for start in range(len(values) - length):
        inputs.append(values[start : start + length])
        targets.append(values[start + 1 : start + length + 1])
    if not inputs:
        raise ValueError("The synthetic stream must exceed the sequence length")
    return torch.tensor(inputs), torch.tensor(targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tiny_dendritron_v1.3_cpu.pt"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/cpu_smoke_v1.3.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.vocab_size < 32:
        raise ValueError("steps, batch size, and vocabulary size must be positive")
    torch.manual_seed(13)
    config = tiny_smoke_config(args.vocab_size)
    model = DendritronLM(config).to(args.device)

    motif = [7, 18, 29, 40, 51, 62, 73, 84, 95, 106, 117, 4]
    values = (motif * 20) + [7]
    input_rows, target_rows = windows(values, args.sequence_length)
    input_rows = input_rows.to(args.device)
    target_rows = target_rows.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        indices = torch.randint(
            0,
            input_rows.shape[0],
            (args.batch_size,),
            device=args.device,
        )
        batch = input_rows.index_select(0, indices)
        targets = target_rows.index_select(0, indices)
        output = model(batch)
        loss = rank_margin_loss(
            output.logits,
            targets,
            margin=0.2,
            hard_negatives=16,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            accuracy = (output.logits.argmax(dim=-1) == targets).float().mean()
        row = {
            "step": step + 1,
            "loss": float(loss.detach()),
            "next_token_accuracy": float(accuracy),
        }
        if step == 0 or step + 1 == args.steps or (step + 1) % 10 == 0:
            history.append(row)
            print(json.dumps(row))

    latency_input = input_rows[:1]
    model.eval()
    with torch.no_grad():
        for _ in range(10):
            model(latency_input)
        timings = []
        for _ in range(50):
            start = perf_counter()
            final = model(latency_input)
            timings.append((perf_counter() - start) * 1000.0)
    timings.sort()
    report = {
        "runtime": torch.__version__,
        "device": str(args.device),
        "cuda_available": bool(torch.cuda.is_available()),
        "tests_expected": 61,
        "steps": args.steps,
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "final_next_token_accuracy": history[-1]["next_token_accuracy"],
        "median_forward_ms": timings[len(timings) // 2],
        "p95_forward_ms": timings[int(0.95 * (len(timings) - 1))],
        "scores_finite": bool(torch.isfinite(final.logits).all()),
        "deep_loop_alpha": config.deep_loop_alpha,
        "deep_loop_beta": config.deep_loop_beta,
        "capacity": model.capacity_ledger().as_dict(),
        "history": history,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "validation_mode": "synthetic-token CPU smoke",
            "history": history,
        },
        args.output,
    )
    reloaded, _ = DendritronLM.from_checkpoint(
        str(args.output),
        map_location=args.device,
    )
    reloaded.eval()
    with torch.no_grad():
        reloaded_scores = reloaded(latency_input).logits
    report["checkpoint_reload_finite"] = bool(
        torch.isfinite(reloaded_scores).all()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved={args.output}")
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
