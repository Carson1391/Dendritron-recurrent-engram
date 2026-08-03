# Dendritron v1.0 Validation

Date: 2026-07-31

## Result

- 50/50 tests passed.
- 10/10 CPU tensor tests passed with CPU PyTorch 2.7.1.
- Python syntax compilation passed for the complete package and build scripts.
- The master specification and runbook rendered through Pandoc successfully.
- Markdown code fences and display-math delimiters are balanced.

## Tokenizer and memory-address contracts tested

- Raw Qwen IDs stay separate from canonical memory IDs.
- Engram normalization merges case variants.
- Engram normalization merges leading-space variants.
- Engram normalization merges accent variants after NFD accent stripping.
- Space, tab, CR, and LF runs share the whitespace canonical key.
- Comma and period remain distinct canonical punctuation classes.
- Canonical projection fingerprints are embedded and verified by JTD.
- Frozen trigram, bigram, and all-sense dictionary priority remains 3 -> 2 -> 1.
- Hash-Engram uses canonical IDs on frozen donor misses.

## Punctuation contracts tested

- `tree bark,` retrieves the frozen `tree bark` row when the synthetic Qwen
  tokenizer fuses `bark,` into one token.
- `tree, bark` leaves the frozen `tree bark` row inactive.
- The dictionary fallback remains available after intervening punctuation.
- Hash-Engram remains active on that frozen donor miss.
- `camera-based` and `don't` each remain one complete word.
- Stage-1 segment construction resets at punctuation.

## CPU recurrent-core contracts retested

- Exact 25/75 primary sparse-capacity invariant.
- Causal geometric interaction.
- Both physical blocks receive gradients.
- Sparse memory changes live vocabulary scores.
- Recurrent block visit order remains 1, 2, 1, 2 for two rounds.

## Real-data measurement still required

`modal run modal_build_joint_transfer_domain.py` must load the exact Stage-2
Qwen tokenizer snapshot and record Qwen's measured canonical-vocabulary
reduction. The paper's 23.43% remains the reference result for its own 128k
tokenizer until that CPU compile completes.
