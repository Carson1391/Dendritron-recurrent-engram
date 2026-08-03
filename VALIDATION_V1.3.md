# Dendritron v1.3 Validation

Date: 2026-08-01
Runtime: PyTorch 2.7.1+cpu
Device: CPU
CUDA available: false

## Executed results

- 61/61 tests passed.
- 15/15 PyTorch-specific tensor tests passed.
- Python compilation passed for the package, tests, training scripts, and
  Stage-3 through Stage-6 scripts.
- Thirty CPU optimizer steps reduced synthetic-token rank-margin loss from
  `0.206182` to `0.186076`.
- Next-token accuracy increased from `2.34%` to `35.16%`.
- Median tiny forward latency was `15.766 ms`; p95 was `28.120 ms`.
- Vocabulary scores and all four recurrent visits remained finite.
- The saved v1.3 CPU checkpoint reloaded and produced finite scores.
- The primary sparse ledger remained exactly 25% memory / 75% conditional
  compute: `592,896` memory coefficients and `1,778,688` compute coefficients.

## Euclidean HarMax contracts exercised

- Contract-pool distances equal ordinary squared Euclidean distances.
- Future-token changes preserve earlier causal outputs.
- Supported anchors receive positive movement coefficients.
- Opposing anchors receive negative movement coefficients.
- Attraction, repulsion, harmonic residual, and confidence traces remain
  finite.
- Rank loss backpropagates through each physical block's HarMax evidence
  parameters.
- Runtime source contains the locked Euclidean derivative contract.

## DeepLoop contracts exercised

- With `R=2`, `N=4`, `alpha=sqrt(8)`, and `beta=1/sqrt(32)`.
- Both physical blocks execute contraction and compute Post-RMSNorm sublayers.
- `alpha` applies at every sublayer visit.
- `beta` initializes designated residual matrices once.
- Skills, selected expert outputs, and selected Hash-Engram rows receive live
  gradients.

## Output and checkpoint contracts exercised

- Chunked vocabulary scores match explicit Euclidean displacement scores.
- The vocabulary scorer avoids the width-expanded pairwise displacement
  allocation.
- `DendritronLM.from_checkpoint()` reconstructs the config and exact state.

## Remaining real-data executions

- Full Qwen surface-index build on the Modal volume.
- Same-content layer-2 anchor extraction.
- Full layer-8/layer-24/live JTD projection fit.
- Live implementation of Section 10.6 expert-owned semantic branch pools.
- Real-corpus recipient training and language-quality evaluation.
