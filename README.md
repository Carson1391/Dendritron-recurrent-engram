# Dendritron v1.4 — CPU/ARM Reference Build

Dendritron keeps the exact Qwen donor tokenizer while running the live model
on CPU/ARM. Qwen/CUDA is used only for offline asset extraction.

## Locked data path

```text
raw Qwen token ID -> exported Qwen input embedding -> live hidden state
complete word     -> dictionary sense row          -> fixed layer-2 point
bigram/trigram    -> frozen layer-8/layer-24 row   -> learned JTD map
all active points -> ordinary Euclidean HarMax     -> recurrent update
live hidden state -> ordinary Euclidean vocabulary distance
```

Words and IDs are addresses and trace metadata. Each dictionary definition
vector already is a concept location in the canonical 2,048D layer-2 frame;
the word itself is never encoded again to manufacture another concept vector.
Every matching sense remains available as its own point.

Punctuation has three explicit roles:

- Raw Qwen punctuation IDs remain in the live hidden/output stream.
- Canonical punctuation remains in Hash-Engram miss keys.
- Punctuation between complete words resets frozen bigram/trigram windows;
  internal apostrophes and hyphens remain part of a word.

Thus `tree bark,` can retrieve `tree bark`, while `tree, bark` cannot. The
preserved recount and exact row-reuse audit are complete. The reviewed result
keeps 421,332 bigrams and 388,443 trigrams, for 809,775 directly reusable
paired donor rows. It admits 190,225 corrected phrases for delta extraction.

HarMax and vocabulary ranking use ordinary Euclidean distance. There is no
learned bilinear, quadratic, Mahalanobis, or diagonal vocabulary metric in
those runtime paths. JTD and LoRA retain ordinary one-input linear maps.

## Immediate real-data sequence

Keep the completed Stage-1 and Stage-2 roots untouched. The current operation
is the corrected Stage-2 delta migration:

```bash
modal run modal_migrate_punctuation_stage2.py --plan-only
modal run modal_migrate_punctuation_stage2.py
```

The first command recomputes and hard-validates the approved counts without a
GPU. The second command copies reusable rows on CPU, loads Qwen only for plan
shards containing new rows, and resumes fingerprint-matching completed shards.
It writes:

```text
/data/dendritron-stage2-punctuation-v2/
  manifest.json
  migration_plan/
  bigrams/
  trigrams/
```

Then compile and align the corrected assets:

```bash
modal run modal_build_joint_transfer_domain.py \
  --stage2-root /data/dendritron-stage2-punctuation-v2

modal run modal_extract_jtd_latent_assets.py --smoke \
  --stage2-root /data/dendritron-stage2-punctuation-v2

modal run modal_extract_jtd_latent_assets.py \
  --stage2-root /data/dendritron-stage2-punctuation-v2

modal run modal_fit_joint_transfer_domain.py
```

`modal_build_joint_transfer_domain.py` retains its historical name; it builds
the collision-checked surface lookup, not the numerical transfer maps.
`modal_extract_jtd_latent_assets.py` exports the full Qwen input-embedding
matrix—including punctuation—and real layer-2/layer-8/layer-24 anchors. The
CPU fitter learns the layer-8 and layer-24 maps. The 2,048D live map starts as
identity and trains with the recipient unless genuine paired live states are
available.

## CPU verification

The recurrent tensor suite was last executed on 2026-08-07 with PyTorch 2.7.1
and CUDA unavailable. The new delta-migration planner was validated on
2026-08-10 in the dependency-free host.

| Check | Result |
|---|---:|
| Current dependency-free discovery | 50 passed, 14 tensor tests skipped |
| Python syntax compilation | passed |
| New punctuation audit tests | 2/2 passed |
| New Stage-2 delta planner tests | 3/3 passed |
| New real-anchor/JTD artifact tests | 3/3 passed |
| Runtime bilinear/quadratic metric scan | clear |
| Sparse-capacity law | exactly 25% memory / 75% compute |

Run the suite:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

## Source of truth

- `DENDRITRON_MASTER_SPEC.md` — architecture and project ledger
- `NEXT_STEP_AFTER_DICTIONARY.md` — immediate punctuation and JTD handoff
- `CHANGES_v1.3_PUNCTUATION_FRONTEND_JTD.md` — correction record
- `VALIDATION_V1.3.md` — executed CPU validation
- `DENDRITRON_V1.0_TOKENIZER_LOCK.md` — Qwen-ID and punctuation policy

Historical reports remain for traceability and are superseded by the v1.3
runtime and validation record.
