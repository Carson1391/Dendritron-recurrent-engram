# Dendritron v1.3 Change Record

Date: 2026-08-07

## Outcome

Version 1.3 closes the handoff between the completed dictionary, the Qwen
symbol frontend, the frozen phrase banks, and the numerical Joint Transfer
Domain (JTD). It also corrects the n-gram inventory workflow for punctuation
boundaries and removes the remaining learned quadratic runtime metric.

## Punctuation inventory correction

The original Stage-1 counter could advance an n-gram window across punctuation.
The runtime boundary policy already prevents such a frozen phrase from being
activated, but it cannot reclaim a top-500k slot. The correction therefore:

1. preserves the original Stage-1 and Stage-2 roots;
2. writes a full punctuation-aware recount to a separate root;
3. compares old and corrected keys by exact UTF-8 phrase identity;
4. reports reusable, added, and retired rows;
5. reserves Qwen extraction for newly admitted rows only.

Added files:

- `stage3_jtd/compare_punctuation_inventories.py`
- `modal_compare_punctuation_inventory.py`
- `tests/test_punctuation_inventory_audit.py`

`corpus_builder.py` now accepts a narrow alternate output root so the recount
cannot overwrite the completed bank accidentally.

## Qwen symbol frontend

The Qwen tokenizer remains the recipient tokenizer. Token IDs alone carry no
learned numerical content, so the new offline asset job exports Qwen's complete
input-embedding matrix row-for-row. This covers punctuation, whitespace,
subwords, special tokens, and ordinary token pieces.

Added `modal_extract_jtd_latent_assets.py`. It validates the exact donor
revision, exports `qwen_token_embeddings.safetensors`, and records symbol
coverage in its manifest. `DendritronLM.load_token_embedding_artifact()` loads
the artifact with exact vocabulary/width validation.

## Dictionary and phrase alignment

Dictionary `hidden_states[2]` rows remain unchanged as the canonical 2,048D
concept frame. Complete words address sense rows directly. The new asset job
also samples deterministic phrases and writes same-text layer-2/layer-8/
layer-24 anchors. The CPU fitter accepts these sharded Safetensors directly.
`modal_fit_joint_transfer_domain.py` runs that fitter beside the mounted data
volume, so no multi-gigabyte anchor transfer is required.

If live/reference pairs are absent, the production-width live-to-joint map is
initialized as identity and trained with the recipient. Layer-8 and layer-24
maps are fitted from real donor anchors first.

## Euclidean geometry lock

HarMax now computes literal ordinary Euclidean squared distance plus its causal
relative-position term. Vocabulary ranking uses normalized ordinary Euclidean
distance. Removed runtime parameters include:

- `metric_factor`;
- `vocabulary_metric`;
- the HarMax metric floor/configuration path.

One-input projections used by JTD and LoRA remain linear transformations; they
are not pairwise bilinear similarity operators.

## Validation

The full CPU suite passes 61/61 tests with CUDA unavailable. New tests exercise
the punctuation audit, real anchor-root loading, optional-live identity fit,
and full-symbol embedding artifact loading.
