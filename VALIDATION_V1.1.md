# Dendritron v1.1 Validation

> Historical record superseded by `VALIDATION_V1.2.md`.

Date: 2026-08-01

## Result

- 53/53 tests passed.
- 8/8 CPU PyTorch tensor tests passed with PyTorch 2.7.1+cpu.
- Python syntax compilation passed for the package and tests.

## New concept-resolution contracts

- Dictionary selection waits until a causal contextual state exists.
- Block 1 retains the closest sparse sense candidates.
- Block 2 contracts that set to one exact sense row.
- A later context shift preserves the latched sense row.
- Block-2 LNGram concept addresses remain identical on later contraction
  visits.
- The selection path stays softmax-free and exposes sense slots, stable sense
  rows, candidate masks, locality scores, weights, and latch state in runtime
  statistics.

## Existing contracts reverified

- JTD exact 3 -> 2 -> 1 lookup preserves every dictionary sense candidate.
- Punctuation and tokenizer-address rules pass.
- Both recurrent physical blocks execute and backpropagate.
- Sparse memory changes live vocabulary scores.
- The primary sparse-capacity ledger remains exactly 25% memory / 75% compute.

## Real-data job status

The full `modal run modal_build_joint_transfer_domain.py` compilation remains
pending on the real Stage-2 and dictionary assets. The current JTD tests use a
synthetic Qwen-style tokenizer and synthetic memory rows.
