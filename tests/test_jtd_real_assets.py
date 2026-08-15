from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

try:
    import torch
    from safetensors.torch import save_file

    from dendritron.config import tiny_smoke_config
    from dendritron.model import DendritronLM
    from stage3_jtd.fit_joint_transfer_domain import (
        _load_real_anchor_root,
        fit,
    )
except ImportError:  # pragma: no cover - dependency-free discovery host
    torch = None


@unittest.skipIf(torch is None, "PyTorch and Safetensors are required")
class JTDRealAssetTests(unittest.TestCase):
    def _write_anchor(self, root: Path, bank: str, offset: float) -> None:
        path = root / "anchors" / f"{bank}.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        reference = torch.arange(32, dtype=torch.float32).reshape(4, 8) + offset
        save_file(
            {
                "row_indices": torch.arange(4, dtype=torch.int64),
                "layer02": reference.to(torch.bfloat16),
                "layer08": (reference + 0.2).to(torch.bfloat16),
                "layer24": (reference - 0.3).to(torch.bfloat16),
            },
            str(path),
        )

    def test_real_anchor_root_combines_both_engram_orders(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_anchor(root, "bigrams", 0.0)
            self._write_anchor(root, "trigrams", 10.0)
            layer8, layer24, reference = _load_real_anchor_root(root)
            self.assertEqual(tuple(layer8.shape), (8, 8))
            self.assertEqual(tuple(layer24.shape), (8, 8))
            self.assertEqual(tuple(reference.shape), (8, 8))
            self.assertTrue(
                torch.allclose(
                    layer8 - reference,
                    torch.full_like(reference, 0.2),
                    atol=0.06,
                )
            )

    def test_fitter_allows_live_map_to_wait_for_recipient_training(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = np.random.default_rng(7)
            reference = generator.normal(size=(16, 8)).astype(np.float32)
            source8 = reference @ np.diag(np.linspace(0.8, 1.2, 8)).astype(np.float32)
            source24 = reference @ np.diag(np.linspace(1.2, 0.8, 8)).astype(np.float32)
            archive = root / "anchors.npz"
            np.savez(
                archive,
                layer8_source=source8,
                layer8_reference=reference,
                layer24_source=source24,
                layer24_reference=reference,
            )
            output = root / "jtd.pt"
            report = fit(
                Namespace(
                    anchors=archive,
                    anchor_root=None,
                    output=output,
                    live_width=8,
                    steps_per_source=3,
                    batch_size=8,
                    learning_rate=1e-3,
                    neighbor_count=2,
                )
            )
            self.assertEqual(report["rows"]["live"], 0)
            self.assertEqual(
                report["reports"]["live"]["status"],
                "identity_initialized_train_with_recipient",
            )
            checkpoint = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(
                torch.equal(checkpoint["live_to_joint"], torch.eye(8))
            )
            self.assertTrue(
                torch.equal(checkpoint["joint_to_live"], torch.eye(8))
            )

    def test_qwen_symbol_artifact_loads_every_vocab_row(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "symbols.safetensors"
            config = tiny_smoke_config(vocab_size=13)
            model = DendritronLM(config)
            values = torch.linspace(
                -1.0,
                1.0,
                steps=config.vocab_size * config.model_width,
            ).reshape(config.vocab_size, config.model_width)
            save_file(
                {"token_embeddings": values.to(torch.bfloat16)},
                str(path),
            )
            model.load_token_embedding_artifact(path)
            self.assertTrue(
                torch.allclose(
                    model.token_embeddings.weight,
                    values.to(torch.bfloat16).to(model.token_embeddings.weight.dtype),
                )
            )


if __name__ == "__main__":
    unittest.main()
