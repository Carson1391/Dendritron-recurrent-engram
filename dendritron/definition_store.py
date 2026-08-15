"""Sparse CPU access to frozen dictionary sense vectors and metadata."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SenseRowLoader = Callable[[Path, int], Any]


def file_sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _default_row_loader(path: Path, local_row: int) -> Any:
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - runtime environment
        raise RuntimeError(
            "Dictionary payload loading requires safetensors>=0.5"
        ) from error
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"layer02"}:
            raise ValueError(f"{path}: expected one layer02 tensor")
        return handle.get_slice("layer02")[local_row]


@dataclass(frozen=True)
class FrozenDefinitionPayload:
    sense_row: int
    sense_id: str
    word_id: int
    surface: str
    normalized: str
    part_of_speech: str
    definition: str
    source: str
    shard_index: int
    local_row: int
    layer02: Any


class FrozenDefinitionStore:
    """Fetch one selected sense row from disk while the bank stays on CPU."""

    def __init__(
        self,
        bank_root: Path,
        *,
        row_loader: SenseRowLoader | None = None,
        validate_lookup: bool = True,
    ) -> None:
        self.bank_root = Path(bank_root)
        manifest_path = self.bank_root / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(self.manifest.get("hidden_state_layer", -1)) != 2:
            raise ValueError("Dictionary bank must contain Qwen layer-2 rows")
        if int(self.manifest.get("hidden_size", -1)) != 2048:
            raise ValueError("Dictionary bank width must be 2048")
        if str(self.manifest.get("storage_dtype")) != "bfloat16":
            raise ValueError("Dictionary bank storage must be bfloat16")

        lookup_record = self.manifest.get("lookup", {})
        recorded_lookup = Path(str(lookup_record.get("path", "")))
        self.lookup_path = (
            recorded_lookup
            if recorded_lookup.is_file()
            else self.bank_root / recorded_lookup.name
        )
        if not self.lookup_path.is_file():
            raise FileNotFoundError(self.lookup_path)
        if (
            validate_lookup
            and file_sha256(self.lookup_path) != lookup_record.get("sha256")
        ):
            raise ValueError("Dictionary lookup hash differs from its manifest")

        self.total_rows = int(self.manifest["rows"])
        self.shard_size = int(self.manifest["shard_size"])
        self.row_loader = row_loader or _default_row_loader

    def _metadata(self, sense_row: int) -> sqlite3.Row:
        if not 0 <= int(sense_row) < self.total_rows:
            raise IndexError(
                f"sense row {sense_row} falls outside 0..{self.total_rows - 1}"
            )
        connection = sqlite3.connect(str(self.lookup_path))
        connection.row_factory = sqlite3.Row
        try:
            row = connection.execute(
                """
                SELECT
                    sense_row,
                    sense_id,
                    word_id,
                    surface,
                    normalized,
                    part_of_speech,
                    definition,
                    source,
                    shard_index,
                    local_row
                FROM senses
                WHERE sense_row = ?
                """,
                (int(sense_row),),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            raise KeyError(f"Dictionary sense row {sense_row} is absent")
        return row

    def get(self, sense_row: int) -> FrozenDefinitionPayload:
        row = self._metadata(sense_row)
        shard_index = int(row["shard_index"])
        local_row = int(row["local_row"])
        if shard_index != int(sense_row) // self.shard_size:
            raise ValueError("Dictionary lookup shard alignment is inconsistent")
        if local_row != int(sense_row) % self.shard_size:
            raise ValueError("Dictionary lookup local-row alignment is inconsistent")
        shard_path = (
            self.bank_root / "shards" / f"shard-{shard_index:05d}.safetensors"
        )
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        vector = self.row_loader(shard_path, local_row)
        shape = tuple(getattr(vector, "shape", ()))
        if shape != (2048,):
            raise ValueError(
                f"{shard_path}: sense row has shape {shape}; expected (2048,)"
            )
        return FrozenDefinitionPayload(
            sense_row=int(row["sense_row"]),
            sense_id=str(row["sense_id"]),
            word_id=int(row["word_id"]),
            surface=str(row["surface"]),
            normalized=str(row["normalized"]),
            part_of_speech=str(row["part_of_speech"]),
            definition=str(row["definition"]),
            source=str(row["source"]),
            shard_index=shard_index,
            local_row=local_row,
            layer02=vector,
        )
