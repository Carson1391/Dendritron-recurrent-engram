# Dendritron v1.3 Euclidean HarMax and DeepLoop Correction

Date: 2026-08-01

## HarMax geometry

Every causal context, memory, and branch contraction now uses:

\[
d_q^2=\|h-z_q\|_2^2+\lambda_p\delta_q^2+\epsilon^2
\]

\[
\Delta
=n\sum_q(y_q-p_q)\frac{z_q-h}{d_q^2}
\]

The v1.2 `metric_factor`, `metric_floor`, and factored
`L.T @ L` geometry were removed from runtime configuration, source, tests, and
the master specification. HarMax keeps causal top-k pooling, target mass,
distance mass, attraction, repulsion, harmonic residual, and confidence.

## DeepLoop placement

For two physical blocks and `R` rounds:

\[
N=2R,\qquad
\alpha=\sqrt{2N},\qquad
\beta=(8N)^{-1/2}
\]

- `alpha` scales the skip path at both Post-RMSNorm residual sublayers.
- `beta` is applied once during initialization.
- Designated residual matrices include Hash-Engram output projections, LNGram
  value/convolution outputs, shared-skill LoRA factors, universal output
  directions, and expert branch matrices.
- JTD definition locations and source-map reference geometry retain their own
  identity/orthogonal initialization.

## Vocabulary geometry

Vocabulary scores remain negative mean squared Euclidean distances between the
normalized live state and tied token embeddings. The implementation now uses
bounded `torch.cdist` vocabulary chunks, avoiding a four-dimensional
`[batch, sequence, vocabulary, width]` displacement allocation.

## Checkpoint path

- Added `DendritronConfig.from_dict()`.
- Added `DendritronLM.from_checkpoint()`.
- Added `smoke_cpu_core.py` and a reload-verified CPU smoke checkpoint.

## Status boundary

The two-block recurrent substrate, memory fusion, universal/skill/expert
compute hierarchy, Euclidean HarMax context contraction, output geometry, and
fixed 25/75 ledger execute end to end. Section 10.6's expert-owned semantic
branch-pool operators remain the next live runtime module.
