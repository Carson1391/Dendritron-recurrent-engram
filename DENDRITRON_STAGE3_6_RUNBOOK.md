# Dendritron Dictionary, Surface Index, JTD, Shared-LoRA, and LNGram Runbook

Version: 1.3  
Date: 2026-08-07

## Locked result

The existing Stage 2 asset remains unchanged:

- 500,000 bigram rows
- 500,000 trigram rows
- Qwen layer 8 and layer 24
- 2,048D BF16 vectors
- exact UTF-8 phrase keys

Every command in this runbook treats those Stage 2 assets as read-only.

The completed inventory was counted before the punctuation-boundary
correction. The runtime boundary rule prevents a cross-punctuation activation,
but it cannot repair which phrases received the top-500k table slots. Preserve
the original inventory and vectors, produce a punctuation-aware recount under
a separate root, and compare exact UTF-8 keys. Existing Stage-2 vectors are
reused for every overlapping key; only newly admitted corrected phrases need
donor extraction.

Device contract:

| Work | Device |
|---|---|
| Qwen layer-2/layer-8/layer-24 donor extraction | Offline A100 CUDA job |
| Definition download, parsing, coverage, inventory, surface index, and JTD fitting | CPU |
| Dendritron memory lookup, LNGram, experts, loops, and Shared-LoRA | CPU/device-agnostic |

The Qwen donor leaves the runtime after its frozen rows are written.

The recipient still uses Qwen's tokenizer. Its raw token IDs are initialized
from a complete exported Qwen input-embedding matrix, so punctuation,
whitespace, subwords, and special symbols enter the live latent stream with
their learned donor vectors.

The new knowledge path is:

```text
dictionary definition
-> ordered definition-word links
-> Qwen layer-2 readout
-> frozen dictionary sense row
-> contextual LNGram address
```

Runtime memory priority is:

```text
exact three-word donor Engram
otherwise exact two-word donor Engram
otherwise one-word dictionary sense candidates
```

The one-word fallback retains every sense until live context selects one.

The three address spaces are:

```text
raw Qwen token IDs
-> Dendritron embeddings and output vocabulary

raw Qwen token IDs
-> frozen canonical projection P
-> collision-checked surface-index hash bucket
-> exact token-tuple verification
-> frozen layer-8/layer-24 Engram row or dictionary sense rows

canonical Qwen token IDs on a frozen donor miss
-> deterministic multi-head Hash-Engram rows
-> trainable local-pattern memory

live hidden states
-> LNGram projection and hard symbols
-> latent 2/3-gram rows
-> trainable concept memory
```

Raw Qwen IDs, canonical memory IDs, and LNGram latent hashes use separate
namespaces. Punctuation remains in the raw stream and canonical Hash-Engram
keys. Frozen word Engrams reset at punctuation/symbol boundaries.

The dictionary contains every eligible single-word headword from the locked
sources. It uses one row per **sense**, so `bark` has separate tree-covering
and animal-sound rows. Every completed Engram word must have dictionary
coverage before Qwen extraction begins.

## Why layer 2

The dictionary is meant to preserve the words and structure composing a
definition. Its canonical record contains both:

1. One early donor vector from `hidden_states[2]`.
2. Ordered links to every complete word in the definition.

A late semantic summary would make the definition easier to compare as a whole
while hiding more of its lexical construction. Layer 2 plus explicit word
links gives the recurrent model direct access to the construction.

The N-gram bank keeps both existing views:

| Bank value | Runtime role |
|---|---|
| N-gram layer 8 | Early live-state assistance |
| N-gram layer 24 | Composed phrase meaning |
| Dictionary layer 2 | Lexical/compositional word-sense knowledge |

Small recipient projections fuse only the selected N-gram or dictionary rows
into the live state. Every complete bank stays in CPU/disk storage.

## Implemented files

| File | Purpose |
|---|---|
| `dendritron/definition_bank.py` | Stable word/sense schemas, definition parsing, and IDs |
| `stage3_dictionary/definition_sources.py` | Versioned OEWN, Wiktionary, and MeSH parsers, downloads, fingerprints, and coverage measurement |
| `stage3_dictionary/prepare_definition_sources.py` | Local CPU definition-source compiler |
| `modal_prepare_definition_sources.py` | Modal CPU launcher for the source build |
| `DEFINITION_SOURCES.md` | Source, version, license, and selection ledger |
| `stage3_dictionary/build_dictionary_inventory.py` | Finalized source inventory and definition-word graph |
| `modal_extract_definition_states.py` | Resumable Qwen layer-2 donor extraction |
| `dendritron/definition_store.py` | Sparse CPU row retrieval for selected dictionary senses |
| `dendritron/universal_subspace.py` | Optional CPU definition-vector PCA diagnostics |
| `stage4_subspace/build_universal_subspace.py` | Optional CPU definition-vector PCA runner |
| `dendritron/lngram_address.py` | Exact NumPy reference addresses |
| `dendritron/lngram.py` | Trainable PyTorch LNGram with counterfactual gradients |
| `dendritron/shared_skill_subspace.py` | Universal/Shared-LoRA task-update basis |
| `dendritron/retrieval.py` | Longest exact 3→2→1 runtime resolver |
| `dendritron/tokenizer.py` | Locked Qwen contract, official Engram canonical projection, Qwen offset alignment, and punctuation-aware complete-word segmentation |
| `stage3_jtd/build_joint_transfer_domain.py` | Historical command name for the CPU compiler from UTF-8 keys to collision-checked surface addresses |
| `modal_build_joint_transfer_domain.py` | CPU-only Modal launcher for the surface compiler |
| `dendritron/jtd.py` | Immutable surface database compiler and read-only runtime suffix index |
| `dendritron/joint_transfer.py` | Layer-2 definition reference frame plus layer-8/layer-24/live transfer maps |
| `stage3_jtd/compare_punctuation_inventories.py` | Read-only exact-key reuse and delta audit for corrected n-gram inventories |
| `modal_compare_punctuation_inventory.py` | Modal CPU launcher for the punctuation inventory audit |
| `modal_extract_jtd_latent_assets.py` | Offline Qwen export of all token embeddings and real layer-2/layer-8/layer-24 anchors |
| `stage3_jtd/fit_joint_transfer_domain.py` | CPU locality-preserving fitter for real layer-2/layer-8/layer-24 anchors |
| `modal_fit_joint_transfer_domain.py` | Modal CPU launcher for fitting maps against anchor files on the mounted data volume |
| `dendritron/engram_store.py` | Lazy row-aligned layer-8/layer-24 payload retrieval |
| `dendritron/hash_engram.py` | Deterministic multi-head token hashes for trainable Engram misses |
| `dendritron/memory_pipeline.py` | Exact 3→2→1 plus Hash-Engram miss plan |
| `dendritron/expert_graph.py` | Many-to-many knowledge/task/skill expert junctions |
| `stage6_lngram/lngram_smoke.py` | Forward/backward LNGram runtime test |

## Stage 3A: build and measure the definitions

Run the source build:

```bash
modal run modal_prepare_definition_sources.py
```

This CPU job acquires and fingerprints:

- Open English WordNet 2025+;
- English Wiktionary from the locked 2026-07-06 Wiktextract snapshot;
- MeSH 2026 descriptors;
- MeSH 2026 supplementary concepts.

It emits the complete curated single-word dictionary, preserving every
retained sense. It also checks every word in the completed 500k bigram and
500k trigram tables. The report includes type coverage, frequency-weighted
coverage, the 1,000 highest-frequency remaining words, exact sense count, and
the projected BF16 vector-bank size. It loads no donor model and uses CPU
resources only.

The canonical outputs are:

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

Engram-word coverage must reach 100% before freezing the inventory. A reviewed
source-grounded JSONL supplement can close a measured gap:

```bash
modal run modal_prepare_definition_sources.py \
  --supplemental-definitions /data/dicts/reviewed_science.jsonl
```

Then build the full CPU inventory:

```bash
modal run modal_extract_definition_states.py --inventory-only
```

That command validates every canonical source hash and the exact Engram key
hashes, retains every dictionary sense, builds ordered definition-word edges,
and writes the SQLite inventory. It stops before Qwen when Engram-word coverage
falls below 100%. The source manifest becomes part of the immutable inventory
contract.

## Stage 3B: encode the finalized definitions

After accepting the source coverage and inspecting the CPU inventory, run:

```bash
modal run modal_extract_definition_states.py
```

It loads Qwen blocks 0 and 1, extracts `hidden_states[2]` from the final token of
the fixed `Concept:` marker, and writes the complete layer-2 bank. A repeat of
the same command resumes valid shards.

The two explicit rebuild controls have separate scopes:

```bash
# CPU inventory only
modal run modal_extract_definition_states.py \
  --rebuild-inventory \
  --inventory-only

# Layer-2 dictionary vector bank only
modal run modal_extract_definition_states.py --rebuild-bank
```

Rebuild flags are reserved for a deliberate, reviewed source-contract change.
An ordinary mismatch stops while preserving the current inventory and vectors.

Full outputs:

```text
/data/dendritron-stage3-dictionary/
  inventory/
    words.jsonl
    senses.jsonl
    dictionary.sqlite3
    inventory_manifest.json
  bank/
    shards/shard-00000.safetensors
    metadata/shard-00000.jsonl
    lookup.sqlite3
    manifest.json
```

Every tensor shard contains only:

```text
layer02: [rows, 2048] BF16
```

The vector storage cost is:

\[
N_{\text{senses}}\times2048\times2\text{ bytes}
\]

At 200,000 senses, that is about 0.82 GB before metadata.

At one million senses, the BF16 vectors occupy about 3.81 GiB. At two million,
they occupy about 7.63 GiB. `dictionary_storage_report.json` gives the exact
count and projected size before Qwen starts.

The live model loads none of that bank as one tensor. `FrozenDefinitionStore`
uses the CPU SQLite index and fetches one selected 2,048D row from the relevant
disk shard. One active definition vector is 4,096 bytes.

## Optional definition-vector diagnostic

`stage4_subspace/build_universal_subspace.py` remains available as a CPU-only
PCA diagnostic. It is outside the required Dendritron build. The model's
16-direction/98%-variance hypothesis applies to successful task-adapter
factors in the Universal/Shared-LoRA weight-space subspace.

## Stage 3C: LNGram

The implemented default uses:

```text
model width          2048
bits per route       4
routes               512
symbol alphabet      16
orders               2 and 3
route memory width   4
readout              signed geometric normalization
```

Table sizes:

| Order | Rows | BF16 value parameters |
|---|---:|---:|
| 2 | 131,072 | 524,288 |
| 3 | 2,097,152 | 8,388,608 |

The forward path is exact:

```text
hidden state
-> RMSNorm
-> address projection
-> hard four-bit symbols
-> exact route-partitioned 2/3-gram table rows
-> signed geometric readout
-> causal depthwise convolution
-> near-zero residual gate
```

The routing projection receives the paper's one-bit counterfactual gradient.
The selected memory rows and readout projections receive ordinary gradients.

Run the PyTorch smoke test in the training environment:

```bash
python stage6_lngram/lngram_smoke.py --device cpu
```

The test requires:

- finite forward values
- a nonzero address-projection gradient
- nonzero gradients in both lookup tables
- correct 2/3-gram prefix masks

## Stage 4A: correct the punctuation inventory and migrate Stage 2

Run the corrected 200M-word count under a new root:

```bash
modal run corpus_builder.py \
  --output-root /data/dendritron-stage1-punctuation-v2 \
  --force
modal run modal_compare_punctuation_inventory.py
```

The comparison is read-only with respect to the completed Stage-2 bank. The
reviewed audit found 809,775 reusable rows and 190,225 new rows. Build the
corrected bank with:

```bash
modal run modal_migrate_punctuation_stage2.py --plan-only
modal run modal_migrate_punctuation_stage2.py
```

`--plan-only` hard-gates the exact reviewed bigram/trigram counts and every
source fingerprint on CPU. The full command copies paired layer-8/layer-24
rows by exact UTF-8 key and extracts only the new rows. It is shard-resumable
and writes `/data/dendritron-stage2-punctuation-v2` while preserving the
original Stage-2 root.

## Stage 4B: lock Qwen, build P, and compile the surface index

The first recipient uses the exact tokenizer revision recorded by the
completed Stage-2 donor manifest:

```text
Qwen/Qwen3.6-35B-A3B
resolved Stage-2 commit
add_special_tokens=False
```

The raw tokenizer stream remains the live input/output alphabet. The compiler
builds a frozen memory-only projection:

```text
decode vocabulary item
-> NFKC -> NFD -> strip accents -> lowercase
-> collapse whitespace runs -> preserve whitespace-only key -> strip
-> canonical memory ID
```

The paper's 23.43% result belongs to its 128k tokenizer. The compiler measures
Qwen directly and records `raw_vocab_size`, `effective_vocab_size`, and
`reduction_percent` with the full projection fingerprint.

Punctuation policy:

```text
raw Qwen punctuation -> hidden states and output vocabulary
canonical punctuation -> Hash-Engram suffixes
punctuation between words -> frozen word-Engram boundary
internal apostrophe/hyphen -> remains inside one word
```

Fast Qwen offsets attach a word match to its actual token position. Therefore
`tree bark,` retrieves `tree bark` even if the final token contains the comma;
`tree, bark` keeps that bigram inactive.

Compile the CPU index after the dictionary inventory and corrected Stage-2
root are final:

```bash
modal run modal_build_joint_transfer_domain.py \
  --stage2-root /data/dendritron-stage2-punctuation-v2
```

This command:

1. Reads the Stage-2 manifest, 500,000 bigram keys, 500,000 trigram keys, and
   finalized dictionary SQLite inventory.
2. Loads the Qwen tokenizer at the Stage-2 resolved revision.
3. Saves tokenizer artifact hashes and BOS/internal examples.
4. Builds and fingerprints the complete raw-to-canonical projection.
5. Measures Qwen's effective-vocabulary reduction.
6. Compiles every exact surface key into both projected token boundary modes
   and a canonical complete-word address.
7. Hashes each address into a BLAKE2b-128 lookup bucket.
8. Retains the complete canonical tuple for collision verification.
9. Preserves `bank_name`, `word_order`, and the original immutable row index.
10. Writes deterministic surface-address collisions and reports cryptographic
   bucket collisions.
11. Records the canonical Hash-Engram and punctuation contracts for donor misses.

Outputs:

```text
/data/dendritron-stage4-jtd/
  tokenizer_snapshot/
  tokenizer_contract.json
  canonical_token_projection.json
  surface_index.sqlite3
  surface_index_collisions.jsonl
  surface_index_manifest.json
```

The full donor vector shards stay in their current locations. The runtime
loader follows the selected bigram/trigram row to its original shard and
returns both `layer08` and `layer24`.

Matching completed surface-index inputs reuse the current artifact. A source fingerprint
change requires the explicit CPU-only `--rebuild` control:

```bash
modal run modal_build_joint_transfer_domain.py --rebuild
```

## Stage 4C: export Qwen symbols and real latent anchors

After the corrected Stage-2 manifest passes review:

```bash
modal run modal_extract_jtd_latent_assets.py --smoke \
  --stage2-root /data/dendritron-stage2-punctuation-v2
modal run modal_extract_jtd_latent_assets.py \
  --stage2-root /data/dendritron-stage2-punctuation-v2
modal run modal_fit_joint_transfer_domain.py
```

This writes the complete row-for-row Qwen input-embedding table and
deterministic phrase anchors with layer-2, layer-8, and layer-24 views. Run the
CPU map fitter against its `--anchor-root`. The live 2,048D map is
identity-initialized and trained with the recipient unless real paired live
states are supplied.

## Stage 4D: runtime address order

At every Qwen token position and complete-word endpoint:

```text
exact trigram donor hit
-> retrieve frozen trigram row

otherwise exact bigram donor hit
-> retrieve frozen bigram row

otherwise one-word dictionary candidates
-> retain every sense

and, on every frozen donor miss
-> hash canonical Qwen-ID suffixes, including punctuation classes

after an initial live state exists
-> compute separate LNGram latent addresses
```

## Dictionary knowledge versus the Universal/Shared-LoRA subspace

These are separate systems:

| System | Built from | Stores |
|---|---|---|
| Dictionary/Engram memory | Source definitions and frozen Qwen donor rows | Exact knowledge payloads on CPU/disk |
| Universal/Shared-LoRA subspace | Successful task LoRA factors | 16-32 reusable weight-space directions |

The Share-style skill implementation is in
`dendritron/shared_skill_subspace.py`. It stacks successful LoRA factors,
extracts principal directions by SVD, assigns fast task coefficients, and
measures the orthogonal `x` residual used for later skill consolidation.

Experts remain a separate graph layer. An expert links dictionary/Engram IDs
and LNGram concepts, a task relation, useful principal skill directions, and
branch specifications. It may reference a successful coefficient prior while
remaining distinct from that prior.

## Validation completed here

Local tests cover:

- separate senses for polysemous words
- dictionary senses beyond the completed Engram vocabulary
- ordered definition-word retention
- one-row CPU/disk definition payload retrieval
- fixed readout marker
- collision-free route partitioning
- exact LNGram table sizes
- optional definition-vector PCA recovery on known low-rank data
- Share-style adapter reconstruction
- orthogonality of the skill `x` residual
- exact 3→2→1 memory priority with variable recipient-token spans
- preservation of all dictionary senses at the one-word fallback
- Qwen-style BOS/internal token compilation
- collision-checked surface hash lookup with exact tuple verification
- deterministic Hash-Engram multi-head addresses on frozen donor misses
- recovery of both immutable Engram vector views from the original shard row
- rejection of mismatched tokenizer fingerprints
- many-to-many expert/skill adjacency and coefficient-prior separation

The local suite now contains 61 passing CPU tests. It includes punctuation
inventory comparison, full-symbol artifact loading, real-anchor archive
loading, optional-live identity initialization, and ordinary Euclidean HarMax
and vocabulary geometry. The real punctuation recount, corrected Stage-2
migration, Qwen symbol/anchor export, and full-bank JTD fit remain the remote
data validations.

## Next connection

1. Recount Stage 1 with punctuation resets under the punctuation-v2 root.
2. Compare corrected and completed keys; review reuse and delta counts.
3. Copy exact-overlap Stage-2 rows and extract only newly admitted phrases.
4. Compile the corrected Qwen surface index and inspect its collision report.
5. Export the full Qwen input-embedding table and real phrase anchor archive.
6. Fit layer-8 and layer-24 maps into the unchanged layer-2 definition frame.
7. Connect real symbol, phrase, dictionary, Hash-Engram, and LNGram paths to the
   CPU-verified recurrent core.
8. Train the identity-initialized live map and recurrent parameters.
9. Connect expert branch execution over the Universal/Shared-LoRA basis.
