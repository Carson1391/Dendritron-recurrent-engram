"""Lazy row-aligned access to the immutable Stage-2 Engram payloads."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


TensorShard = Mapping[str, Any]
ShardLoader = Callable[[Path], TensorShard]


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _default_safetensor_loader(path: Path) -> TensorShard:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - runtime environment
        raise RuntimeError(
            "Engram payload loading requires safetensors>=0.5"
        ) from error
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            key: handle.get_tensor(key)
            for key in handle.keys()
        }


@dataclass(frozen=True)
class FrozenEngramPayload:
    bank_name: str
    word_order: int
    row_index: int
    shard_index: int
    local_row: int
    layer08: Any
    layer24: Any


class FrozenEngramStore:
    """Load layer-8/layer-24 rows without changing or duplicating the bank."""

    def __init__(
        self,
        stage2_root: Path,
        *,
        loader: ShardLoader | None = None,
        maximum_cached_shards: int = 2,
        validate_shard_on_first_load: bool = True,
    ) -> None:
        if maximum_cached_shards < 1:
            raise ValueError("maximum_cached_shards must be positive")
        self.stage2_root = Path(stage2_root)
        manifest_path = self.stage2_root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not bool(self.manifest.get("completed")):
            raise ValueError("Stage-2 Engram manifest is incomplete")
        states = self.manifest.get("states", {})
        if tuple(states.get("hidden_state_layers", ())) != (8, 24):
            raise ValueError("Stage-2 Engram bank must contain layers 8 and 24")
        if int(states.get("hidden_size", -1)) != 2048:
            raise ValueError("Stage-2 Engram hidden size must be 2048")

        self.shard_size = int(self.manifest["sharding"]["shard_size"])
        self.loader = loader or _default_safetensor_loader
        self.maximum_cached_shards = int(maximum_cached_shards)
        self.validate_shard_on_first_load = bool(validate_shard_on_first_load)
        self._validated_paths: set[Path] = set()
        self._cache: OrderedDict[tuple[str, int], TensorShard] = OrderedDict()

    @staticmethod
    def _bank_name(word_order: int) -> str:
        if word_order == 2:
            return "bigrams"
        if word_order == 3:
            return "trigrams"
        raise ValueError("Frozen donor Engrams have word order 2 or 3")

    def _table(self, bank_name: str) -> Mapping[str, Any]:
        table = self.manifest.get("tables", {}).get(bank_name)
        if not isinstance(table, Mapping):
            raise ValueError(f"Stage-2 manifest has no {bank_name} table")
        return table

    def _shard_artifact(
        self,
        bank_name: str,
        shard_index: int,
    ) -> tuple[Path, Mapping[str, Any]]:
        table = self._table(bank_name)
        artifacts = table.get("tensor_files", ())
        if not 0 <= shard_index < len(artifacts):
            raise IndexError(f"{bank_name} shard {shard_index} is unavailable")
        artifact = artifacts[shard_index]
        recorded = Path(str(artifact["path"]))
        path = (
            recorded
            if recorded.is_file()
            else self.stage2_root
            / bank_name
            / "shards"
            / recorded.name
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, artifact

    def _load_shard(self, bank_name: str, shard_index: int) -> TensorShard:
        key = (bank_name, shard_index)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        path, artifact = self._shard_artifact(bank_name, shard_index)
        if self.validate_shard_on_first_load and path not in self._validated_paths:
            if path.stat().st_size != int(artifact["bytes"]):
                raise ValueError(f"Engram shard size mismatch: {path}")
            if file_sha256(path) != str(artifact["sha256"]):
                raise ValueError(f"Engram shard hash mismatch: {path}")
            self._validated_paths.add(path)

        shard = self.loader(path)
        if set(shard) != {"layer08", "layer24"}:
            raise ValueError(
                f"{path}: expected layer08 and layer24 tensors, found {sorted(shard)}"
            )
        self._cache[key] = shard
        self._cache.move_to_end(key)
        while len(self._cache) > self.maximum_cached_shards:
            self._cache.popitem(last=False)
        return shard

    def get(self, *, word_order: int, row_index: int) -> FrozenEngramPayload:
        bank_name = self._bank_name(word_order)
        table = self._table(bank_name)
        rows = int(table["rows"])
        if not 0 <= row_index < rows:
            raise IndexError(
                f"{bank_name} row {row_index} falls outside 0..{rows - 1}"
            )
        shard_index, local_row = divmod(int(row_index), self.shard_size)
        shard = self._load_shard(bank_name, shard_index)
        return FrozenEngramPayload(
            bank_name=bank_name,
            word_order=word_order,
            row_index=int(row_index),
            shard_index=shard_index,
            local_row=local_row,
            layer08=shard["layer08"][local_row],
            layer24=shard["layer24"][local_row],
        )

    def validate_all_shards(self) -> dict[str, int]:
        counts = {"bigrams": 0, "trigrams": 0}
        for bank_name in counts:
            table = self._table(bank_name)
            for shard_index in range(len(table.get("tensor_files", ()))):
                path, artifact = self._shard_artifact(bank_name, shard_index)
                if path.stat().st_size != int(artifact["bytes"]):
                    raise ValueError(f"Engram shard size mismatch: {path}")
                if file_sha256(path) != str(artifact["sha256"]):
                    raise ValueError(f"Engram shard hash mismatch: {path}")
                self._validated_paths.add(path)
                counts[bank_name] += 1
        return counts
