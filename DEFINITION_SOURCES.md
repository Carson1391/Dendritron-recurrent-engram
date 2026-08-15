# Dendritron Definition Sources

The dictionary bank is built from source-grounded definitions before any Qwen
vectors are extracted.

| Source | Locked version | Purpose | Terms |
|---|---:|---|---|
| Open English WordNet Plus | 2025+ | Curated general English senses, polysemy, and validated proper nouns | WordNet License + CC BY 4.0 |
| English Wiktionary via Wiktextract | 2026-07-06 dump | Broad lexical, scientific, and mathematical vocabulary | CC BY-SA 4.0 + GFDL |
| Medical Subject Headings descriptors | 2026 | Biomedical and life-science terminology | NLM MeSH Terms and Conditions |
| Medical Subject Headings supplementary concepts | 2026 | Chemicals, diseases, organisms, protocols, and emerging biomedical terms | NLM MeSH Terms and Conditions |

Every eligible English single-word headword and every retained sense enters
the bank. Each sense preserves its source name, source version, source sense
key, exact definition text, and licensing metadata. This is the complete
one-word dictionary fallback after trigram and bigram Engram lookup.

Run the CPU source build:

```bash
modal run modal_prepare_definition_sources.py
```

It creates:

```text
/data/dendritron-stage3-definition-sources/
  raw/
  canonical/
    open_english_wordnet_plus-2025+.jsonl
    english_wiktionary-2026-07-06.jsonl
    mesh_descriptors-2026.jsonl
    mesh_supplementary_concepts-2026.jsonl
    definition_sources_manifest.json
    coverage_report.json
    dictionary_storage_report.json
```

The coverage report measures every unique word and frequency-weighted word
occurrence in the completed 500k bigram and 500k trigram key files. Engram-word
coverage must reach 100% before freezing the source contract. Reviewed
source-grounded supplements can be added with:

```bash
modal run modal_prepare_definition_sources.py \
  --supplemental-definitions /data/dicts/reviewed_science.jsonl
```

The storage report gives the exact sense count and projected BF16 vector bytes
before Qwen extraction begins. One layer-2 row costs 4,096 bytes. The
dictionary remains in CPU/disk shards and Dendritron fetches selected rows.

CUDA is used only by the offline Qwen extraction that creates those frozen
rows. Dendritron itself uses CPU/device-agnostic lookup and computation.
