# Dendritron v1.1 Concept-Resolution Repair

> Superseded by v1.2. The selector/latch described below was removed after the
> JTD address layer and latent transfer domain were separated correctly.

Date: 2026-08-01

## Repaired junction

The JTD compiler remains the collision-checked exact address bridge. It emits
all dictionary sense rows for a one-word fallback. The live model now performs
the contextual concept decision after that lookup:

```text
JTD exact 3 -> 2 -> 1 lookup
-> all one-word sense candidates
-> first causal contextual interaction
-> Block 1 locality-ranked candidate expansion
-> Block 2 exact single-sense contraction
-> latched sense row and Block-2 LNGram addresses
-> later recurrent visits
```

## Code changes

- `MemoryPayloads` carries stable `definition_sense_rows` beside definition
  vectors and masks.
- Initial token embeddings receive phrase/hash memory while dictionary senses
  wait for the first contextual block interaction.
- Dictionary locality uses reciprocal squared unit-vector distance. The result
  stays positive and softmax-free.
- Block 1 keeps up to `definition_candidate_top_k` candidate senses.
- Block 2 uses an exact one-hot forward selection with a normalized
  straight-through locality gradient for the learned recipient projection.
- The selected sense row remains fixed for all later visits in that forward
  episode.
- Block 2 records the LNGram addresses formed from the contracted sense and
  reuses those addresses on later Block-2 visits.

## Status boundary

- Contextual sense selection and latching: implemented and CPU tensor tested.
- JTD compiler: implemented and synthetic integration tested.
- Full Qwen JTD artifact: pending the real CPU Modal compilation.
- Full-bank recipient projection fitting: pending the real-data training run.
