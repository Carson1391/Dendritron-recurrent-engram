"""Run a deterministic forward/backward smoke test of Dendritron LNGram.

Example:
    python stage6_lngram/lngram_smoke.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run(device: str) -> dict[str, object]:
    import torch

    from dendritron.lngram import LNGramMemory

    torch.manual_seed(7)
    model = LNGramMemory(
        model_width=32,
        bits_per_route=4,
        orders=(2, 3),
        route_memory_width=4,
        readout_mode="distance",
    ).to(device)
    hidden = torch.randn(2, 7, 32, device=device, requires_grad=True)
    output, stats = model(hidden, return_stats=True)
    loss = output.square().mean()
    loss.backward()

    address_gradient = model.address_projection.weight.grad
    table_gradients = {
        order: float(model.tables[str(order)].grad.norm().item())
        for order in model.orders
    }
    result = {
        "output_shape": list(output.shape),
        "loss": float(loss.item()),
        "route_count": model.route_count,
        "alphabet_size": model.alphabet_size,
        "table_rows": model.table_row_counts,
        "address_projection_gradient_norm": float(address_gradient.norm().item()),
        "table_gradient_norms": table_gradients,
        "valid_positions": {
            order: int(stats.valid[order].any(dim=-1).sum().item())
            for order in model.orders
        },
        "passed": bool(
            torch.isfinite(output).all()
            and torch.isfinite(address_gradient).all()
            and address_gradient.abs().sum() > 0
            and all(value > 0 for value in table_gradients.values())
        ),
    }
    if not result["passed"]:
        raise RuntimeError(f"LNGram smoke test failed: {result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
