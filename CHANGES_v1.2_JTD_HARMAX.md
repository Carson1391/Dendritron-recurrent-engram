# Dendritron v1.2 JTD and HarMax Correction

> Superseded by v1.3 for HarMax distance geometry and DeepLoop beta placement.
> The v1.2 JTD boundary remains active.

Date: 2026-08-01

## Corrected JTD boundary

The numerical JTD and the surface address index are separate mechanisms:

```text
surface word / token IDs / word ID / sense ID
-> row lookup and trace metadata

frozen layer-2 definition vector
-> unchanged canonical concept location

layer-8 Engram / layer-24 Engram / live state
-> source-specific learned map
-> layer-2 joint concept frame
```

`dendritron/joint_transfer.py` implements that reference frame and its source
maps. `stage3_jtd/fit_joint_transfer_domain.py` fits maps from same-content
layer-2 anchors with point alignment and neighborhood preservation.

The v1.1 Block-1 candidate ranking, Block-2 winner selection, sense-row latch,
and forced LNGram concept-address reuse were removed from `memory_fusion.py`
and `recurrent_core.py`. Every addressed definition sense remains present as
its own joint-space point. Continuous inverse-distance mass determines current
influence and is recomputed on every recurrent visit.

The historical `modal_build_joint_transfer_domain.py` command now identifies
its output accurately as a surface index:

- `surface_index.sqlite3`
- `surface_index_collisions.jsonl`
- `surface_index_manifest.json`

## HarMax replacement

`dendritron/geometric_attention.py` now implements the locked derivative:

\[
\Delta
=n\sum_q(y_q-p_q)
\frac{G(z_q-h)}{d_q^2},
\qquad
G=L^\top L+\lambda I
\]

The runtime reports distance mass, target mass, signed coefficients,
attraction mass, repulsion mass, harmonic residual, and confidence. The old
learned pairwise context score and separate content-transport matrix were
removed.

LNGram fusion and vocabulary scoring now use Euclidean distances. Linear
source maps, LoRA factors, and one-input learned transformations remain part
of the architecture.

## Recurrent and gradient repairs

- Every physical block now executes two sequential Post-RMSNorm residual
  sublayers.
- DeepLoop `alpha` applies at both sublayers.
- DeepLoop `beta` initializes the HarMax metric factor rather than multiplying
  every residual visit.
- Skill output factors, expert branch outputs, and nested Hash-Engram gates
  begin near zero with live gradients.
- Definition vectors never receive word strings or identifier channels.

## Real-data boundary

The code path and synthetic/tiny CPU tensors are verified. The completed Qwen
layer-8/layer-24 and definition banks remain reusable. Full surface-index
compilation and full-bank JTD projection fitting remain data jobs on the real
assets.
