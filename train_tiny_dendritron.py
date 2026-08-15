"""Overfit a tiny Qwen-token corpus through the two-block Dendritron path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dendritron.config import tiny_smoke_config
from dendritron.model import DendritronLM
from dendritron.output_geometry import rank_margin_loss
from dendritron.tokenizer import (
    LOCKED_QWEN_TOKENIZER_ID,
    build_canonical_token_projection,
    build_tokenizer_contract,
)


DEFAULT_TEXT = (
    "memory anchors meaning. compute changes the thought. "
    "two blocks carry thought through repeated rounds. "
    "skills are shared operations. experts solve specialized relations. "
)


def windows(values: list[int], length: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(values) <= length:
        raise ValueError("The corpus must contain more tokens than the window")
    inputs = []
    targets = []
    for start in range(len(values) - length):
        inputs.append(values[start : start + length])
        targets.append(values[start + 1 : start + length + 1])
    return torch.tensor(inputs), torch.tensor(targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--tokenizer", default=LOCKED_QWEN_TOKENIZER_ID)
    parser.add_argument(
        "--tokenizer-revision",
        required=True,
        help="Exact resolved tokenizer commit recorded by the Stage-2 manifest.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("/models/huggingface"))
    parser.add_argument(
        "--device",
        default="cpu",
        help="Reference training device. CPU is the canonical Dendritron target.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/tiny_dendritron.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install transformers and tokenizers for the Qwen smoke run") from error
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        cache_dir=args.cache_dir,
        trust_remote_code=False,
        use_fast=True,
    )
    tokenizer_contract = build_tokenizer_contract(
        tokenizer,
        tokenizer_id=args.tokenizer,
        requested_revision=args.tokenizer_revision,
        resolved_revision=args.tokenizer_revision,
    )
    projection = build_canonical_token_projection(tokenizer)
    config = tiny_smoke_config(len(tokenizer))
    model = DendritronLM(config).to(args.device)
    values = list(tokenizer.encode(DEFAULT_TEXT, add_special_tokens=False))
    if tokenizer.eos_token_id is not None:
        values.append(int(tokenizer.eos_token_id))
    input_rows, target_rows = windows(
        values,
        args.sequence_length,
    )
    input_rows = input_rows.to(args.device)
    target_rows = target_rows.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = []
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
        if step == 0 or (step + 1) % 25 == 0:
            with torch.no_grad():
                accuracy = (
                    output.logits.argmax(dim=-1) == targets
                ).float().mean()
            row = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "next_token_accuracy": float(accuracy),
            }
            history.append(row)
            print(json.dumps(row))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.to_dict(),
            "tokenizer_contract": tokenizer_contract.to_record(),
            "canonical_token_projection": projection.to_record(),
            "history": history,
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
