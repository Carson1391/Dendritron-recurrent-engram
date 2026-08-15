# Dendritron Stage 3-6 v0.7

This release restores the complete dictionary and the CPU-first Dendritron
device contract.

Changes:

- Compiles every eligible English single-word headword from OEWN 2025+, English
  Wiktionary, and MeSH.
- Retains every distinct source sense and one layer-2 row per sense.
- Requires 100% coverage of all words occurring in the completed 500k bigram
  and 500k trigram banks before Qwen extraction.
- Supports reviewed source-grounded supplemental JSONL definitions for any
  measured Engram-word coverage gap.
- Emits the exact sense count and projected BF16 vector-bank storage before
  Qwen starts.
- Keeps definition words as ordered word-ID graph links into the full
  dictionary.
- Adds sparse `FrozenDefinitionStore` CPU/disk row retrieval.
- Confines CUDA to offline Qwen donor extraction.
- Restores the 16-32 principal directions to the Universal/Shared-LoRA
  weight-space subspace built from successful task adapters.
- Moves dictionary-vector PCA out of the required model path and keeps it as
  an optional CPU diagnostic.
- Preserves every completed Stage-1 and Stage-2 artifact.

CPU source build:

```bash
modal run modal_prepare_definition_sources.py
```

Offline Qwen layer-2 extraction begins after the full dictionary reaches 100%
Engram-word coverage and its exact storage report is accepted.
