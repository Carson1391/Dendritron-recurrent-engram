from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dendritron.engram_store import FrozenEngramStore


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenEngramStoreTests(unittest.TestCase):
    def test_row_maps_to_original_bank_shard_and_both_layers(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = {}
            for bank_name, shard_names in {
                "bigrams": ("shard-00000.safetensors", "shard-00001.safetensors"),
                "trigrams": ("shard-00000.safetensors",),
            }.items():
                shard_root = root / bank_name / "shards"
                shard_root.mkdir(parents=True)
                rows = []
                for name in shard_names:
                    path = shard_root / name
                    path.write_bytes((bank_name + name).encode("utf-8"))
                    rows.append(
                        {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": sha256(path),
                        }
                    )
                artifacts[bank_name] = rows

            manifest = {
                "completed": True,
                "states": {
                    "hidden_state_layers": [8, 24],
                    "hidden_size": 2048,
                },
                "sharding": {"shard_size": 2},
                "tables": {
                    "bigrams": {
                        "rows": 3,
                        "tensor_files": artifacts["bigrams"],
                    },
                    "trigrams": {
                        "rows": 2,
                        "tensor_files": artifacts["trigrams"],
                    },
                },
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            loaded = []

            def loader(path):
                loaded.append(path.name)
                shard_index = int(path.stem.split("-")[-1])
                base = shard_index * 2
                return {
                    "layer08": [[base + 0, 8], [base + 1, 8]],
                    "layer24": [[base + 0, 24], [base + 1, 24]],
                }

            store = FrozenEngramStore(root, loader=loader)
            payload = store.get(word_order=2, row_index=2)
            self.assertEqual(payload.bank_name, "bigrams")
            self.assertEqual(payload.shard_index, 1)
            self.assertEqual(payload.local_row, 0)
            self.assertEqual(payload.layer08, [2, 8])
            self.assertEqual(payload.layer24, [2, 24])
            self.assertEqual(loaded, ["shard-00001.safetensors"])

            repeated = store.get(word_order=2, row_index=3 - 1)
            self.assertEqual(repeated.layer24, [2, 24])
            self.assertEqual(loaded, ["shard-00001.safetensors"])


if __name__ == "__main__":
    unittest.main()
