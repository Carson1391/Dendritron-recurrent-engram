# Dendritron v0.8 Architecture Lock - Superseded

Date superseded: 2026-07-31

The v0.8 byte-frontend experiment is retained only in `CHANGES_v0.8.md` and
its historical checkpoint metadata.

The active architecture is defined by:

- `DENDRITRON_MASTER_SPEC.md` version 1.3;
- `DENDRITRON_V1.0_TOKENIZER_LOCK.md`;
- `CHANGES_v1.3_EUCLIDEAN_HARMAX.md`;
- `DENDRITRON_STAGE3_6_RUNBOOK.md`.

The live recipient uses the exact Stage-2 Qwen tokenizer for raw input/output
IDs. A frozen Engram-style canonical projection supplies memory-only IDs.
Punctuation remains in the live and Hash-Engram streams and forms a boundary
for frozen complete-word n-grams.

The active contraction geometry is the ordinary Euclidean HarMax derivative.
DeepLoop alpha scales both recurrent residual visits, and beta initializes the
designated residual-branch matrices once.
