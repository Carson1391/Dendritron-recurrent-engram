# Next Step After the Dictionary

Date: 2026-08-07

## Short answer

The dictionary layer-2 hidden states already are the canonical concept points.
A surface word does not need another bare-word Qwen pass. At runtime, the Qwen
tokenizer and complete-word alignment identify the word; the dictionary lookup
returns every matching sense row directly.

Before fitting phrase rows into that space, correct the punctuation-sensitive
n-gram inventory.

## What maps where

| Runtime item | Address mechanism | Numerical value in the joint space |
|---|---|---|
| Raw token, punctuation, whitespace, subword, special token | Exact Qwen token ID | Exported Qwen input-embedding row initializes the live token vector |
| Complete word | Normalized word lookup plus sense metadata | One immutable dictionary layer-2 vector per sense |
| Exact bigram | Boundary-aware Qwen surface lookup | Frozen layer-8 and layer-24 rows mapped into the layer-2 frame |
| Exact trigram | Boundary-aware Qwen surface lookup | Frozen layer-8 and layer-24 rows mapped into the layer-2 frame |
| Frozen phrase miss | Canonical Qwen-ID Hash-Engram address | Trainable local-pattern memory |
| Context after live state exists | LNGram hidden-state address | Trainable latent concept memory |

Words, token IDs, and sense IDs locate and trace records; they are not extra
concept vectors. Context changes the Euclidean influence of retrieved sense
points rather than moving the frozen dictionary rows.

## Punctuation rule

- Keep raw punctuation in the Qwen input/output stream.
- Keep canonical punctuation in Hash-Engram keys.
- Reset frozen word bigram/trigram windows whenever punctuation or symbol
  material occurs between words.
- Allow trailing punctuation after the last word: `tree bark,` still retrieves
  `tree bark`.
- Keep internal apostrophes and hyphens inside a word.

The original Stage-1 counter crossed punctuation. A runtime barrier stops bad
activations, but the original top-500k lists can still contain wasted rows.
Therefore the inventory must be recounted before choosing the canonical phrase
rows for JTD fitting.

## Completed punctuation audit

| Table | Reusable paired rows | New donor rows | Retired rows |
|---|---:|---:|---:|
| Bigrams | 421,332 | 78,668 | 78,668 |
| Trigrams | 388,443 | 111,557 | 111,557 |
| Total | 809,775 | 190,225 | 190,225 |

## Completed

The punctuation-aware Stage-2 migration completed with 809,775 copied rows and
190,225 newly extracted rows. The corrected bank is complete at
`/data/dendritron-stage2-punctuation-v2`; the original bank remains preserved.

## Run this now

```bash
modal run modal_build_joint_transfer_domain.py
```

The launcher now requires the reviewed punctuation-v2 manifest and writes the
surface index under `/data/dendritron-stage4-jtd-punctuation-v2`.

## Then

1. Export the complete Qwen input-embedding matrix and the same-text
   layer-2/layer-8/layer-24 phrase anchors.
2. Fit the layer-8 and layer-24 maps into the unchanged layer-2 frame.
3. Load the surface index, token embeddings, definition bank, corrected phrase
   bank, and fitted maps into Dendritron.
4. Train the identity-initialized live map with the recipient.

The surface-index result is the current gate.
