# Dendritron Stage 3-6 v0.6

The definition stage now begins with real lexical sources and a measured
coverage report.

Changes:

- Added Open English WordNet 2025 ingestion.
- Added English Wiktionary 2026-07-06 Wiktextract ingestion.
- Added MeSH 2026 descriptor and supplementary-concept ingestion.
- Added exact source versions, licenses, attributions, raw hashes, canonical
  hashes, and source sense keys.
- Added resumable raw downloads and reusable per-source canonical artifacts.
- Added a hard stop when Kaikki's moving Wiktionary file advances beyond the
  locked dump date.
- Added coverage measurement against every word in the completed Stage 2
  bigram and trigram tables.
- Added a highest-frequency missing-word report to guide any specialist source
  addition.
- Made the finalized definition-source manifest mandatory for the dictionary
  inventory.
- Removed the old placeholder `science.jsonl` and `math.jsonl` workflow.
- Kept completed Stage 1 and Stage 2 assets immutable.

The next command is CPU/network-only:

```bash
modal run modal_prepare_definition_sources.py
```

Qwen dictionary extraction begins only after the resulting source and coverage
manifests are reviewed.
