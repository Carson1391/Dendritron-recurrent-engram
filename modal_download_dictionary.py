"""Download the layer-2 dictionary bank from Modal volume to local disk.

Lists /data/dendritron-stage3-dictionary/ on the dendritron-corpus volume,
then downloads every file into ./dendritron-stage3-dictionary/ locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

VOLUME_NAME = "dendritron-corpus"
VOLUME_MOUNT = "/data"
REMOTE_DICT = Path(VOLUME_MOUNT) / "dendritron-stage3-dictionary"
LOCAL_ROOT = Path(__file__).parent / "dendritron-stage3-dictionary"

app = modal.App("dendritron-download-dictionary")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)


@app.function(
    cpu=1.0,
    memory=2048,
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: volume},
)
def list_remote() -> list[str]:
    """List every file under the dictionary root on the volume."""
    root = Path(REMOTE_DICT)
    if not root.exists():
        return [f"ERROR: {REMOTE_DICT} does not exist on volume {VOLUME_NAME}"]

    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(root))
            size = p.stat().st_size
            files.append(f"{rel}\t{size}")
    return files


@app.function(
    cpu=1.0,
    memory=4096,
    timeout=6 * 60 * 60,
    volumes={VOLUME_MOUNT: volume},
)
def download_file(rel_path: str) -> bytes:
    """Read one file from the volume and return its bytes."""
    full = REMOTE_DICT / rel_path
    if not full.is_file():
        raise FileNotFoundError(f"Remote file not found: {full}")
    return full.read_bytes()


def main() -> None:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    with app.run():
        print(f"Listing {REMOTE_DICT} on volume '{VOLUME_NAME}'...")
        listing = list_remote.remote()

        if not listing:
            print(f"ERROR: {REMOTE_DICT} is empty or does not exist.")
            sys.exit(1)

        if listing[0].startswith("ERROR:"):
            print(listing[0])
            sys.exit(1)

        entries = []
        for line in listing:
            parts = line.split("\t")
            rel = parts[0]
            size = int(parts[1]) if len(parts) > 1 else 0
            entries.append((rel, size))

        total_bytes = sum(s for _, s in entries)
        print(f"Found {len(entries)} files, total {total_bytes / 1e6:.1f} MB")

        for i, (rel, size) in enumerate(entries):
            local_path = LOCAL_ROOT / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if local_path.exists() and local_path.stat().st_size == size:
                print(f"[{i+1}/{len(entries)}] SKIP {rel} ({size} bytes, already exists)")
                continue

            print(f"[{i+1}/{len(entries)}] Downloading {rel} ({size} bytes)...", flush=True)
            data = download_file.remote(rel)
            local_path.write_bytes(data)
            print(f"  -> {local_path} ({len(data)} bytes)")

        print(f"\nDone. Downloaded to {LOCAL_ROOT}")


if __name__ == "__main__":
    main()
