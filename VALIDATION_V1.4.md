# Dendritron v1.4 Validation

Date: 2026-08-10

## Executed locally

| Validation | Result |
|---|---:|
| Dependency-free unit discovery | 50 passed, 14 tensor tests skipped |
| New Stage-2 delta planner tests | 3/3 passed |
| Python syntax compilation | passed |
| Exact-key cross-shard rank movement | passed |
| Audit/plan mismatch stop | passed |
| Accent-sensitive UTF-8 identity | passed |

Command:

```bash
python -m unittest discover -s tests -v
```

The skipped tests require PyTorch/Safetensors in the execution environment.
The Modal image declares those dependencies and remains the execution host for
the real tensor migration and Qwen delta extraction.

## Hard launch gates

The paid donor function starts only after all of these agree:

- original completed Stage-2 manifest and shard hashes;
- corrected bigram and trigram inventory hashes;
- punctuation audit hash and counts;
- recomputed exact-key migration counts;
- reviewed counts of 809,775 reusable and 190,225 new rows;
- locked Qwen model revision, hidden states 8/24, width 2,048, BF16, and
  final-non-padding-token pooling.

## Remote operation pending

- CPU `--plan-only` run against the Modal volume;
- corrected row materialization;
- Qwen extraction for 190,225 new phrases;
- final million-row manifest and shard-hash verification.

