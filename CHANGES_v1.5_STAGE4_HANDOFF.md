# Dendritron v1.5 Stage-4 Handoff

Date: 2026-08-11

## Outcome

The corrected punctuation-aware Stage-2 Engram bank is complete:

- 500,000 bigrams;
- 500,000 trigrams;
- 809,775 paired layer-8/layer-24 rows copied by exact phrase identity;
- 190,225 paired rows freshly extracted from the locked Qwen donor.

## Safety correction

Stage-4 launchers now default to
`/data/dendritron-stage2-punctuation-v2` and write new artifacts beneath
`/data/dendritron-stage4-jtd-punctuation-v2`.

Before compiling a surface index or extracting latent anchors, the code
requires the reviewed contract: punctuation-aware addressing, 500,000 rows per
order, paired `layer08`/`layer24` tensors, hidden-state layers `[8, 24]`, and the
exact 809,775/190,225 migration totals.

This change updates data routing and validation only. Dictionary layer-2
vectors remain the fixed concept reference frame. Surface words and IDs remain
lookup metadata. Runtime geometry remains Euclidean.

## Next command

```bash
modal run modal_build_joint_transfer_domain.py
```

Expected gates include 500,000 bigram rows, 500,000 trigram rows, the complete
dictionary sense count, and zero cryptographic hash collisions.
