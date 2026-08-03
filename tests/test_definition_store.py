from __future__ import annotations

import hashlib
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from dendritron.definition_store import FrozenDefinitionStore


class FrozenDefinitionStoreTests(unittest.TestCase):
    def test_one_cpu_row_is_loaded_for_the_selected_sense(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard_root = root / "shards"
            shard_root.mkdir()
            shard_path = shard_root / "shard-00000.safetensors"
            shard_path.write_bytes(b"fixture")

            lookup = root / "lookup.sqlite3"
            connection = sqlite3.connect(str(lookup))
            connection.executescript(
                """
                CREATE TABLE senses (
                    sense_row INTEGER PRIMARY KEY,
                    sense_id TEXT NOT NULL UNIQUE,
                    word_id INTEGER NOT NULL,
                    surface TEXT NOT NULL,
                    normalized TEXT NOT NULL,
                    part_of_speech TEXT NOT NULL,
                    definition TEXT NOT NULL,
                    source TEXT NOT NULL,
                    shard_index INTEGER NOT NULL,
                    local_row INTEGER NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT INTO senses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    0,
                    "sense_bark_tree",
                    7,
                    "bark",
                    "bark",
                    "noun",
                    "The protective outer covering of a tree.",
                    "oewn:2025",
                    0,
                    0,
                ),
            )
            connection.commit()
            connection.close()
            digest = hashlib.sha256(lookup.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "hidden_state_layer": 2,
                        "hidden_size": 2048,
                        "storage_dtype": "bfloat16",
                        "rows": 1,
                        "shard_size": 50_000,
                        "lookup": {
                            "path": str(lookup),
                            "sha256": digest,
                        },
                    }
                ),
                encoding="utf-8",
            )
            loads = []

            def load_one(path: Path, local_row: int) -> np.ndarray:
                loads.append((path, local_row))
                return np.zeros(2048, dtype=np.float16)

            store = FrozenDefinitionStore(root, row_loader=load_one)
            payload = store.get(0)

        self.assertEqual(payload.sense_id, "sense_bark_tree")
        self.assertEqual(payload.definition, "The protective outer covering of a tree.")
        self.assertEqual(loads, [(shard_path, 0)])
        self.assertEqual(payload.layer02.shape, (2048,))


if __name__ == "__main__":
    unittest.main()
