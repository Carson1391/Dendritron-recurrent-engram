# Dendritron v1.3 Validation

Date: 2026-08-07

## Environment

- Python: 3.12
- PyTorch: 2.7.1
- NumPy: 2.3.5
- Safetensors: 0.5.3
- `CUDA_VISIBLE_DEVICES=''`
- `torch.cuda.is_available() == False`

## Executed results

| Validation | Result |
|---|---:|
| Full unit discovery | 61/61 passed |
| Punctuation inventory audit tests | 2/2 passed |
| Real JTD asset tests | 3/3 passed |
| Python syntax compilation | passed |
| Runtime learned pairwise-metric source scan | clear |
| Device used by tensor tests | CPU |

Command:

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

## New contracts exercised

- Old and corrected bigram/trigram inventories are compared without mutating
  either input.
- Exact UTF-8 overlap determines reusable donor rows.
- Added and retired rows are emitted separately for both orders.
- Real Safetensors anchor roots concatenate bigram and trigram layer-2,
  layer-8, and layer-24 views with aligned rows.
- Missing live pairs produce an identity-initialized 2,048D live map.
- Nonmatching live width requires genuine paired live/reference states.
- Full Qwen token-embedding artifacts load only when vocabulary size and model
  width match the recipient exactly.
- HarMax has no learned metric matrix and uses ordinary Euclidean displacement.
- Vocabulary scores have no learned diagonal metric.

## Existing contracts retained

- Exact 25% memory / 75% compute sparse-capacity allocation.
- Separate 500k bigram and 500k trigram namespaces.
- Dictionary definition vectors pass through JTD unchanged.
- Every retrieved dictionary sense remains an independent point.
- Punctuation-safe longest 3→2→1 lookup.
- Hash-Engram punctuation retention on frozen donor misses.
- Two physical recurrent blocks with two residual sublayers per visit.
- CPU backward gradients through the recurrent, memory, expert, skill, and
  output paths.

## Remote data validation still pending

- Full punctuation-aware 200M-word recount.
- Old/new inventory reuse and delta report on the Modal volume.
- Corrected Stage-2 row migration plus donor extraction for the delta.
- Full Qwen symbol-table and phrase-anchor export.
- Full CPU fit of the layer-8 and layer-24 JTD maps.

These are data-producing runs, not untested architectural decisions. The next
gate is the punctuation inventory audit.
