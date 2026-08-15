# Dendritron v1.4 Stage-2 Delta Handoff

Date: 2026-08-10

## Outcome

The punctuation recount is complete. The corrected Stage-1 inventories retain
809,775 of the original one million phrase keys and admit 190,225 new keys.
Version 1.4 adds the missing executable path that turns that reviewed split
into a corrected paired layer-8/layer-24 Stage-2 bank.

## Reviewed migration counts

| Table | Reusable | Extract new | Retire old |
|---|---:|---:|---:|
| Bigrams | 421,332 | 78,668 | 78,668 |
| Trigrams | 388,443 | 111,557 | 111,557 |
| Total | 809,775 | 190,225 | 190,225 |

## Added implementation

- `stage2_delta/build_migration_plan.py`
  - recomputes the exact UTF-8 overlap from source files;
  - requires the computed, audited, and reviewed counts to agree;
  - writes one corrected-rank plan file per destination shard;
  - records each reusable row's original table, shard, and local row;
  - fingerprints every input and plan shard.
- `modal_migrate_punctuation_stage2.py`
  - preserves `/data/dendritron-stage2` as immutable input;
  - writes `/data/dendritron-stage2-punctuation-v2`;
  - copies both donor views together for every reusable phrase;
  - materializes copy-only shards on CPU;
  - loads Qwen only for shards containing newly admitted phrases;
  - extracts only the 190,225 reviewed delta rows;
  - validates and resumes completed destination shards;
  - builds corrected `keys.jsonl`, SQLite lookup tables, and final manifest.
- `tests/test_stage2_delta_migration.py`
  - exercises rank movement across source shards;
  - exercises mixed reusable/extraction destinations;
  - verifies exact UTF-8 identity;
  - verifies audit mismatch stops before materialization.

## Run order

```bash
modal run modal_migrate_punctuation_stage2.py --plan-only
modal run modal_migrate_punctuation_stage2.py
```

The resulting manifest is directly compatible with the existing surface-index
compiler and latent-anchor exporter through:

```bash
modal run modal_build_joint_transfer_domain.py \
  --stage2-root /data/dendritron-stage2-punctuation-v2
```

