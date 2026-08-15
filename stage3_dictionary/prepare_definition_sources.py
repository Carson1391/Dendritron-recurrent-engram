"""Acquire and compile Dendritron's canonical definition sources.

This is CPU and network work.  It performs no donor-model inference.

Example:

    python stage3_dictionary/prepare_definition_sources.py \
        --raw-root /data/dendritron-stage3-definition-sources/raw \
        --canonical-root /data/dendritron-stage3-definition-sources/canonical \
        --ngram-keys /data/dendritron-stage2/bigrams/keys.jsonl \
        --ngram-keys /data/dendritron-stage2/trigrams/keys.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stage3_dictionary.definition_sources import prepare_definition_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument(
        "--ngram-keys",
        type=Path,
        action="append",
        required=True,
        help="Completed Stage 2 keys.jsonl file; repeatable",
    )
    parser.add_argument(
        "--supplemental-definitions",
        type=Path,
        action="append",
        default=[],
        help="Reviewed source-grounded JSONL definitions; repeatable",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in args.ngram_keys:
        if not path.is_file():
            raise FileNotFoundError(f"Stage 2 key file is unavailable: {path}")
    result = prepare_definition_sources(
        raw_root=args.raw_root,
        canonical_root=args.canonical_root,
        ngram_paths=args.ngram_keys,
        supplemental_paths=args.supplemental_definitions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
