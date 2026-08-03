# Dendritron v1.2 Validation

Date: 2026-08-01
Runtime: PyTorch 2.7.1+cpu
Device: CPU

## Executed results

- 56/56 tests passed.
- 11/11 PyTorch tensor tests passed.
- Python compilation passed for the package, tests, and Stage-3 JTD scripts.
- Thirty CPU optimizer steps reduced tiny rank-margin loss from `0.207046` to
  `0.174178`.
- Final vocabulary scores were finite.
- All four recurrent visits reported finite HarMax residuals.
- Each physical-block visit executed two residual sublayers.
- The primary sparse-capacity ledger remained exactly 25% memory / 75%
  conditional compute.

## JTD contracts exercised

- Definition vectors pass through the joint reference function by exact tensor
  identity.
- Words and sense-row IDs remain metadata.
- All retrieved sense points retain positive continuous locality mass.
- Locality weights sum to one at active positions.
- The source/reference locality loss reaches zero for matching geometry.
- Block-owned dictionary sense selection and latching symbols are absent from
  runtime source.

## HarMax contracts exercised

- Future-token changes leave earlier causal outputs unchanged.
- Explicit supported anchors receive positive movement coefficients.
- Explicit opposing anchors receive negative movement coefficients.
- Attraction, repulsion, residual, and confidence traces remain finite.
- Rank loss backpropagates through each physical block's metric factor.
- Runtime package code contains zero `nn.Bilinear` operators and zero instances
  of the revoked `SignedGeometricAttention` class.

## Gradient contracts exercised

- Shared-skill output factors receive gradients.
- Selected high-dimensional expert branch outputs receive gradients.
- Selected Hash-Engram rows receive gradients.
- Sparse memory changes live vocabulary scores.

## Pending real-data executions

- Full Qwen surface-index build on the Modal volume.
- Same-content layer-2 anchor extraction.
- Full layer-8/layer-24/live JTD projection fit.
- Real-corpus recipient training and language-quality evaluation.
