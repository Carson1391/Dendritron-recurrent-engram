# DENDRITRON
## Master Build Specification and Project Ledger

Version: 1.3  
Date: 2026-08-07  
Purpose: one authoritative map of the architecture, completed assets, remaining work, interfaces, and tests.

---

> **v1.0 tokenizer lock:** `DENDRITRON_V1.0_TOKENIZER_LOCK.md` supersedes the
> byte frontend in the v0.8 runtime lock. The live recipient uses the exact
> Stage-2 Qwen tokenizer for raw input/output IDs and a frozen Engram-style
> canonical projection only for conditional-memory addresses. CPU/ARM remains
> the live target; Qwen weights remain confined to offline donor extraction.

> **v0.9 geometry lock:** the HarMax derivative contraction field supersedes
> the v0.8 bilinear geometric mixer. Expert-owned branch specifications now
> bind explicit reasoning operators to HarMax contraction pools, and every
> physical block executes two sequential DeepLoop residual sublayers. Block 1
> expands candidate structure; Block 2 contracts and verifies it.

> **v1.2 JTD correction:** frozen Qwen layer-2 definition vectors are the
> canonical 2,048D concept frame and remain unchanged. Surface words and IDs
> perform lookup only. Separate learned maps place layer-8, layer-24, and live
> states into the definition frame. The temporary Block-1/Block-2 dictionary
> sense selector and concept latch are revoked.

> **v1.3 frontend, punctuation, and Euclidean lock:** the donor input-embedding
> table seeds every raw Qwen symbol, including punctuation. Complete words use
> exact dictionary-sense lookup. The original Stage-1 inventory receives a
> punctuation-aware recount and row-reuse audit before becoming canonical.
> HarMax and vocabulary ranking use ordinary Euclidean distance; the remaining
> learned quadratic metric has been removed.

---

## 1. How to Use This Document

Every architectural item receives one of six statuses:

| Status | Meaning |
|---|---|
| **LOCKED** | Part of the intended architecture. Changes require an explicit decision-log entry. |
| **CODED** | Executable code exists and local structural tests pass; its full data or training run remains pending. |
| **IMPLEMENTED** | Code or data artifact exists and has passed its current validation. |
| **NEXT** | The next bounded deliverable. |
| **OPEN** | A real design choice that still needs an experiment or explicit decision. |
| **REVOKED** | Historical decision retained for traceability and superseded by a later locked decision. |

Every subsystem must state:

1. Purpose
2. Inputs
3. Outputs
4. Frozen and trainable state
5. Acceptance test

This file is the source of truth. Scripts implement it; conversations explain or revise it.

---

## 2. Architecture in One Sentence

Dendritron uses the exact Qwen donor tokenizer, applies longest 3→2→1
collision-checked lookup over layer-8/layer-24 phrase Engrams and layer-2
dictionary senses, covers exact donor misses with trainable token-hashed Engram
memory, uses a separate LNGram hidden-state address space to connect active
concepts and tasks to the 16-32 principal directions of its
Universal/Shared-LoRA weight-space subspace through many-to-many expert
junctions, and repeatedly executes expert-owned branches through two stored
HarMax geometric blocks while episode-local coefficients learn the current
problem.

---

## 3. Locked Architectural Decisions

| ID | Decision | Status |
|---|---|---|
| D-001 | The first recipient uses the tokenizer from `Qwen/Qwen3.6-35B-A3B` at the exact resolved revision recorded by the completed Stage-2 donor manifest. The tokenizer supplies IDs; the donor input-embedding table supplies the recipient's learned symbol initialization; the same raw IDs define the output vocabulary. | **LOCKED** |
| D-002 | The model stores two ordered physical geometric blocks and reuses them for \(R\) rounds. Stored depth is 2; effective thought depth is \(2R\). | **LOCKED** |
| D-003 | **REVOKED:** geometric attention uses bilinear similarity, sparse selection, and signed normalization. Replaced by D-032. | **REVOKED** |
| D-004 | Routing, memory fusion, expert selection, and decoding use softmax-free operators. | **LOCKED** |
| D-005 | Bigram and trigram Engrams remain separate memory banks. | **LOCKED** |
| D-006 | Each Engram row stores the full 2,048D BF16 layer-8 vector and the full 2,048D BF16 layer-24 vector from the donor. | **LOCKED** |
| D-007 | Donor vectors come from the final non-padding token of each isolated phrase. | **LOCKED** |
| D-008 | Exact UTF-8 phrase text remains the permanent cross-tokenizer identity for each Engram row. | **LOCKED** |
| D-009 | A surface-memory index compiles phrase identities into boundary-aware suffix addresses using the locked Qwen tokenizer revision and frozen canonical token projection. This address index remains separate from the latent Joint Transfer Domain. | **LOCKED** |
| D-010 | LNGram supplies learned discrete concept addresses derived from hidden states. Dendritron uses these addresses to connect concept records, skill nodes, expert adjacency, and experiential memory. | **LOCKED** |
| D-011 | The Shared-LoRA skill subspace begins with 16 principal factor directions and can grow toward 32 when persistent residual evidence supports consolidation. | **LOCKED** |
| D-012 | Principal Shared-LoRA factors are stable skill directions. Episode-local coefficients select and mix a few directions to solve the current task. | **LOCKED** |
| D-013 | An expert is a many-to-many junction linking dictionary/Engram knowledge IDs and LNGram concepts, a requested task relation, useful principal skill directions, and branch specifications. An expert can reference a coefficient prior while remaining distinct from that prior. | **LOCKED** |
| D-014 | Principal skill factors and committed expert junctions remain fixed throughout an episode. Episode-local LoRA coefficients can update at every block visit in learning mode. Successful coefficient traces enter the experience tier and may become initialization priors attached to one or more expert junctions. | **LOCKED** |
| D-015 | DeepLoop's aligned tied-loop rule governs recurrent stability: every residual-sublayer visit uses Post-RMSNorm with \(\alpha=(2N)^{1/2}\), while designated residual matrices receive initialization gain \(\beta=(8N)^{-1/2}\). Visit alignment \(\kappa_R\) is measured as a diagnostic. | **LOCKED** |
| D-016 | Geometric routing compares the current state with skill anchors. Each selected skill returns its adjacent expert IDs through the LNGram/skill graph; runtime performs local expert activation rather than a global expert search. | **LOCKED** |
| D-017 | Branch structure belongs to the expert junction. Its branch specification persists with the expert, while numerical branch activations are recreated from the current state and working coefficients at every physical-block visit. | **LOCKED** |
| D-018 | Frozen donor Engrams and learned experience Engrams occupy separate tiers. Successful coefficient traces and outcome records append to the experience tier while the million donor rows remain immutable. | **LOCKED** |
| D-019 | The dictionary bank stores one immutable row per word sense. Every sense preserves its exact definition, ordered definition-word IDs, and one 2,048D Qwen hidden-state vector from `hidden_states[2]` at the final token of a fixed readout marker. | **LOCKED** |
| D-020 | The dictionary uses a shallow layer-2 donor view so the record preserves lexical/compositional definition structure. Layer-24 dictionary summaries are excluded from the canonical bank. | **LOCKED** |
| D-021 | The dictionary is a complete curated single-word sense bank. Every retained word sense has an exact definition, ordered definition-word IDs, and one frozen layer-2 row. The bank remains on CPU/disk and runtime fetches selected rows sparsely. | **LOCKED** |
| D-022 | Dendritron has one 16-32D Universal/Shared-LoRA subspace in weight space, fitted from successful task-adapter updates. Dictionary vectors remain an independently addressed knowledge bank. | **LOCKED** |
| D-023 | The JTD reference frame is the frozen Qwen layer-2 definition geometry. Definition rows pass through unchanged; separate learned maps align layer-8 Engrams, layer-24 Engrams, and live recipient states to that frame. | **LOCKED** |
| D-024 | Runtime memory resolution uses the longest exact word-order match ending at each position: three-word donor Engram, otherwise two-word donor Engram, otherwise all one-word dictionary sense points. Context interacts with those points continuously in the joint geometry. | **LOCKED** |
| D-025 | The 16-direction/98%-variance claim is tested on successful task-adapter factors in weight space. Persistent task residuals can expand the Universal/Shared-LoRA basis toward 32 directions. | **LOCKED** |
| D-026 | The surface-memory index hashes each boundary-aware canonical Qwen-ID tuple into a compact lookup bucket and retains the complete tuple for exact collision verification. Exact UTF-8 surface text remains the frozen-row identity. | **LOCKED** |
| D-027 | A trainable Hash-Engram path hashes local canonical Qwen-ID suffixes when the frozen bigram/trigram bank misses. Dictionary senses remain available in the exact 3→2→1 route during that miss. | **LOCKED** |
| D-028 | LNGram uses its own learned projection from live hidden states into discrete latent 2/3-gram addresses. Qwen token hashes, frozen donor row addresses, and LNGram latent addresses occupy separate namespaces and tables. | **LOCKED** |
| D-029 | The canonical definition source contract is Open English WordNet 2025+, English Wiktionary’s locked 2026-07-06 Wiktextract snapshot, and MeSH 2026 descriptors plus supplementary concepts. Every eligible English single-word headword and every retained sense enters the one-word bank. Exact source sense IDs, versions, definitions, licenses, raw hashes, and canonical hashes remain recorded. | **LOCKED** |
| D-030 | Definition-source coverage is measured against every word in the completed 500k bigram and 500k trigram tables before any dictionary donor extraction. Engram-word coverage must reach 100%; reviewed source-grounded supplements close any measured gap. | **LOCKED** |
| D-031 | CUDA is confined to offline Qwen donor extraction. Definition preparation, inventory construction, JTD, sparse memory lookup, LNGram, experts, recurrent blocks, and the Universal/Shared-LoRA subspace run on CPU/device-agnostic Dendritron code. | **LOCKED** |
| D-032 | Replacing D-003, geometric attention is the negative derivative of a HarMax cross-entropy over a causally bounded contraction pool. Ordinary Euclidean distance supplies HarMax mass; target mass minus distance mass produces signed attraction/repulsion, and the harmonic residual measures how well the pool explains the current state. | **LOCKED** |
| D-033 | Every physical block contains two sequential Post-RMSNorm residual sublayers: HarMax geometric/memory/branch contraction followed by shared-skill and expert-soma computation. DeepLoop scaling applies to every sublayer visit. | **LOCKED** |
| D-034 | The two physical blocks have complementary recurrent biases. Block 1 expands candidate relations and branch proposals; Block 2 contrasts, verifies, contracts, and integrates them. Both blocks retain the complete branch-operator vocabulary. | **LOCKED** |
| D-035 | One raw Qwen-ID stream supplies live input embeddings and output-vocabulary IDs. A separate frozen surjective projection P supplies canonical IDs only to surface-memory and Hash-Engram addressing. | **LOCKED** |
| D-036 | Projection P follows the official Engram demonstration algorithm: per-token decode, NFKC, NFD, accent stripping, lowercase, whitespace-run collapse, protected whitespace-only key, outer stripping, and raw-piece fallback for U+FFFD. The paper's 23.43% reduction is a reference measurement on its 128k tokenizer; the compiler measures and records Qwen's actual reduction. | **LOCKED** |
| D-037 | Raw punctuation remains in the Qwen hidden-state and output stream. Canonical punctuation remains in Hash-Engram keys. Frozen word Engrams use Qwen offset alignment and treat any non-whitespace material between words as a boundary; internal apostrophes and hyphens remain inside a word. | **LOCKED** |
| D-038 | **REVOKED:** Block 1 ranks dictionary senses, Block 2 selects one row, and later visits latch that row and its LNGram address. Replaced by D-039. | **REVOKED** |
| D-039 | Surface words, token IDs, word IDs, and sense IDs remain lookup/trace metadata. Every retrieved layer-2 definition vector stays at its frozen concept location; continuous joint-space locality supplies context-dependent influence without a block-owned sense selector or latch. | **LOCKED** |
| D-040 | One offline artifact exports the complete Qwen input-embedding matrix row-for-row. Raw punctuation, whitespace, subword, and special-token IDs therefore enter Dendritron through learned donor symbol vectors; complete words additionally address dictionary senses. | **LOCKED** |
| D-041 | HarMax contraction and vocabulary ranking use ordinary Euclidean distances. Learned bilinear, quadratic, Mahalanobis, and diagonal vocabulary metrics are excluded from these runtime paths. One-input JTD and LoRA projections remain ordinary linear transformations. | **LOCKED** |
| D-042 | The pre-correction Stage-1 inventory remains preserved while a second 200M-word recount resets n-gram windows at punctuation/symbol boundaries. Exact UTF-8 overlap determines which Stage-2 rows can be reused and which corrected rows require donor extraction. | **LOCKED** |

---

## 4. Current Project Status

| Stage | Artifact | Status | Evidence |
|---|---|---|---|
| 1 | Science-first 200M-word corpus; top 500k bigrams and top 500k trigrams | **NEXT** | The original inventory is complete; its counter crossed punctuation. `corpus_builder.py` now writes a preserved punctuation-v2 recount, and `modal_compare_punctuation_inventory.py` measures exact row reuse before migration. |
| 2 | Frozen Qwen donor Engrams | **IMPLEMENTED / AUDIT SOURCE** | Separate bigram/trigram tables with full layer-8/layer-24 BF16 vectors are complete and remain the reuse source for the punctuation-v2 row migration. |
| 3S | Canonical definition-source acquisition and coverage | **CODED** | OEWN 2025+, Wiktionary 2026-07-06, and MeSH 2026 streaming parsers; source/license/hash contracts and synthetic parser tests pass. Real CPU source build and coverage report are next. |
| 3A | Dictionary inventory and definition-word graph code | **CODED** | `build_dictionary_inventory.py`; finalized source-manifest verification, synthetic polysemy, and definition-link tests pass. |
| 3B | Layer-2 dictionary donor bank | **IMPLEMENTED (USER-REPORTED)** | The user reports retaining the completed definition/sense hidden-state bank. Local code validates the row/shard contract; the remote manifest is checked by the next latent-asset job. |
| 3C | Optional definition-vector PCA diagnostic | **CODED** | `build_universal_subspace.py`; CPU-only diagnostic excluded from the required model path. |
| 3D | LNGram latent-address module | **CPU VERIFIED** | Exact 2/3-gram route addresses, trainable tables, Euclidean-distance readout, and one-bit counterfactual routing gradients pass CPU tensor tests. |
| 3E | Longest 3→2→1 runtime resolver | **CODED** | `dendritron/retrieval.py`; variable recipient-token spans, trigram/bigram priority, one-word polysemy preservation, and lower-order decomposition access pass local tests. |
| 3F | Expert junction graph | **CODED** | `dendritron/expert_graph.py`; many-to-many concept/task/skill junctions, optional coefficient-prior references, and local adjacency routing pass local tests. |
| 4A | Locked Qwen tokenizer and canonical projection contract | **CODED** | `dendritron/tokenizer.py`; resolves the exact Stage-2 revision, implements the official Engram normalizer, fingerprints projection P, measures Qwen's effective vocabulary, and verifies case/space/accent/whitespace equivalence while retaining distinct punctuation classes. Real Qwen snapshot compilation remains the CPU data run. |
| 4B | Surface-memory compiler and runtime index | **CODED** | `stage3_jtd/build_joint_transfer_domain.py` and `dendritron/jtd.py`; canonical token hashes, exact tuple verification, complete-word offset fallback, tokenizer/projection fingerprints, punctuation barriers, and 3→2→1 runtime lookup pass synthetic integration tests. The historical filename remains for compatibility. |
| 4C | Frozen Engram payload loader | **CODED** | `dendritron/engram_store.py`; maps the selected row back to the immutable layer-8/layer-24 shard with hash validation and lazy caching. |
| 4C-D | Frozen definition payload loader | **CODED** | `dendritron/definition_store.py`; fetches addressed layer-2 sense rows from CPU/disk with row-alignment and lookup-hash validation. |
| 4D | Trainable Hash-Engram miss addressor | **CODED** | `dendritron/hash_engram.py` and `dendritron/memory_pipeline.py`; deterministic multi-head canonical-Qwen-ID hashes retain punctuation and activate on frozen donor misses while dictionary candidates survive. |
| 4E | Layer-2 joint concept frame | **CPU VERIFIED** | `dendritron/joint_transfer.py` and `dendritron/memory_fusion.py`; definitions pass through bit-for-bit, every retrieved sense remains a separate point, and continuous Euclidean locality supplies live influence. The rejected selector/latch symbols are absent from runtime source. |
| 4F | Qwen symbol seed and real JTD anchors | **CODED** | `modal_extract_jtd_latent_assets.py` exports every learned Qwen input symbol, including punctuation, and row-aligned layer-2/layer-8/layer-24 phrase anchors. Full Modal execution follows the punctuation inventory decision. |
| 5 | JTD and sparse memory-fusion maps | **CPU VERIFIED / REAL FIT PENDING** | Definition points remain identity. Layer-8/layer-24 maps fit from real sharded anchors; the live map starts as identity at width 2,048 and trains with the recipient. Synthetic CPU fitting and artifact loading pass. |
| 7 | Two-block HarMax recurrent core | **CPU VERIFIED** | Both stored blocks execute two sequential Post-RMSNorm sublayers and backpropagate through HarMax target/distance mass, attraction/repulsion, and harmonic residuals. |
| 8 | Skill anchors, working LoRA state, expert graph, and expert-owned branches | **OPEN** | Ownership, operator vocabulary, branch specification, HarMax pool, and soma equations are locked; the live HarMax branch path remains to be implemented and measured. |
| 9 | DeepLoop training and adaptive reasoning | **CODED** | Fixed recurrent rounds and adaptive relative-change stopping execute; full loop-alignment measurements remain pending. |
| 10 | Softmax-free vocabulary objective and decoding | **CPU VERIFIED** | Tied Euclidean vocabulary distances, rank-margin loss, greedy decoding, and backward gradients pass CPU tests. |

“Implemented” records a completed and validated artifact. “Coded” records a
real executable implementation whose data-producing or PyTorch runtime run is
still pending.

---

## 5. Offline Assets Already Built

### 5.1 Phrase inventory

- 500,000 content-filtered word bigrams
- 500,000 content-filtered word trigrams
- Exact UTF-8 text
- Raw corpus frequency
- Separate rank order for each n-gram order

These are **word** n-grams. Their runtime recipient-token lengths are variable.

### 5.2 Frozen donor values

For phrase \(s_i\):

\[
e_i^{(8)} =
F_G(\tau_G(s_i))_{\text{last}}^{(8)}
\in \mathbb{R}^{2048}
\]

\[
e_i^{(24)} =
F_G(\tau_G(s_i))_{\text{last}}^{(24)}
\in \mathbb{R}^{2048}
\]

Both values belong to the same Engram row. The 2-gram and 3-gram banks have independent row spaces and lookup tables.

### 5.3 Frozen versus trainable

| Object | State |
|---|---|
| Exact phrase text and frequency | Frozen |
| Donor layer-8 and layer-24 vectors | Frozen |
| Donor model | Offline-only |
| Recipient phrase-to-token mapping | Deterministically compiled |
| Recipient memory projections | Trainable |
| LNGram addressing projection | Trainable during learning windows |
| Principal skill factors | Fixed within an episode; revised only at consolidation |
| Episode-local LoRA coefficients | Fast trainable state |
| Committed expert knowledge/task/skill junctions and branch specifications | Frozen after commit |
| Skill-to-expert adjacency | Appendable after successful episodes |
| Coefficient priors and experience Engram tier | Appendable |

### 5.4 Dictionary knowledge bank

The dictionary bank is a separate frozen knowledge asset. It uses one row per
word sense:

```text
sense_row
word_id
exact definition text
ordered definition_word_ids[]
Qwen layer02[2048]
```

The definition text is source-grounded before Qwen sees it:

```text
Open English WordNet 2025
English Wiktionary 2026-07-06
MeSH 2026 descriptors and supplementary concepts
-> exact single-word senses
-> coverage measurement against Stage 2 Engram vocabulary
-> frozen definition-source manifest
-> Qwen layer-2 encoding
```

Qwen supplies the shallow latent representation. It does not generate the
definitions. Each row retains the exact lexical source version and sense key;
the source manifest retains licenses, attributions, URLs, and hashes.

For a definition \(d_s\), append the same fixed readout marker to every row and
extract:

\[
e_s^{(2)}
=
F_G(\tau_G(d_s \Vert \text{marker}))_{\text{last}}^{(2)}
\in\mathbb{R}^{2048}
\]

The exact word sequence remains graph-addressable. For example:

```text
bark/tree sense
  -> tough
  -> protective
  -> outer
  -> covering
  -> tree
```

Polysemous words own several sense rows. LNGram and current context choose among
those rows during live execution.

---

## 6. Tokenizer, Surface Addressing, and Joint Transfer

### 6.1 One raw Qwen stream

The first recipient locks the tokenizer from `Qwen/Qwen3.6-35B-A3B` at the
exact resolved Stage-2 revision. Raw IDs perform three jobs:

1. Index the recipient's learned starting embeddings.
2. Preserve punctuation, case, spacing, and exact model symbols in the hidden-state stream.
3. Define the tied output vocabulary used to generate the next token.

Qwen tokenization runs on CPU/ARM. Qwen model weights remain offline donor
machinery. The two recurrent blocks evolve hidden states rather than token
IDs; raw IDs only establish and decode the external symbol sequence.

### 6.2 Frozen canonical projection for memory only

A parallel copy of each raw ID is projected before surface-memory and
Hash-Engram lookup:

\[
x_t' = P(x_t), \qquad P:\mathcal V_{Qwen}\twoheadrightarrow\mathcal V_M
\]

The precomputed projection follows the official Engram demonstration:

```text
decode one vocabulary ID
-> NFKC
-> NFD
-> strip Unicode accents
-> lowercase
-> collapse [space, tab, CR, LF] runs to one space
-> protect a whitespace-only canonical key
-> strip outer whitespace
-> raw tokenizer-piece fallback when decoding contains U+FFFD
-> assign one canonical integer per distinct result
```

`P` is a frozen integer lookup table. It changes memory addresses only:

```text
raw Qwen IDs ------------------> token embeddings / output vocabulary
      |
      +-> frozen P -> canonical IDs -> surface index / Hash-Engram
```

The Engram paper measured a 23.43% effective-vocabulary reduction on its own
128k tokenizer. That number motivates the mechanism and is recorded as a
reference. Dendritron's compiler measures the exact Qwen reduction:

\[
r_{Qwen}=1-\frac{|\mathcal V_M|}{|\mathcal V_{Qwen}|}
\]

The manifest stores both vocabulary sizes, `reduction_percent`, the algorithm
name, and a SHA-256 fingerprint of the complete projection. The Qwen value is
accepted only after measurement.

### 6.3 Surface-memory index

Exact UTF-8 phrase text remains the permanent identity between the completed
donor bank and the recipient:

\[
\tau_G(s_i)\rightarrow(e_i^{(8)},e_i^{(24)}),\qquad s_i\rightarrow i
\]

The CPU surface compiler creates two complementary indices for the same immutable
row \(i\):

1. A fast variable-length canonical-Qwen-ID suffix address.
2. A canonical complete-word address used with live Qwen offset mappings.

The second path handles tokenizer pieces that fuse punctuation with a word.
Both paths preserve `bank_name`, `word_order`, row index, exact surface text,
and frequency.

### 6.4 Joint latent domain

The frozen Qwen layer-2 definition bank already supplies the canonical 2,048D
concept geometry:

\[
d_s \longmapsto d_s
\]

where each sense vector remains at its extracted layer-2 location. Exact
surface words, token IDs, word IDs, and sense IDs stay outside the vector as
lookup and trace metadata. Three source-specific maps align the remaining
views:

\[
J_8e^{(8)},\qquad J_{24}e^{(24)},\qquad J_hh
\]

into the same definition frame. `dendritron/joint_transfer.py` implements the
maps and `stage3_jtd/fit_joint_transfer_domain.py` fits them from same-content
layer-2 anchor pairs with point alignment plus locality preservation.

### 6.5 End-gram addressing

At position \(t\), the fast path queries n-grams ending at that position:

\[
m_i^R=P(\tau_R(s_i,\text{boundary}))
=(x'_{i,1},\ldots,x'_{i,L_i})
\]

A source-word bigram can occupy two, three, five, or more Qwen tokens. Source
word order and Qwen-token span therefore remain separate fields. BOS and
ordinary internal-text encodings are compiled before projection; the complete
canonical tuple remains stored for collision verification.

### 6.6 Punctuation and complete-word alignment

The raw Qwen sequence keeps every punctuation token. Hash-Engram receives the
canonical form of those tokens, so punctuation continues to contribute local
syntactic memory. Frozen phrase Engrams and dictionary rows follow a stricter
complete-word policy:

- Qwen's fast offset mapping aligns Unicode word spans to actual token positions.
- Any non-whitespace material between adjacent words starts a new frozen n-gram segment.
- Apostrophes and hyphens surrounded by word characters remain inside one word.
- A frozen match is attached to the token position overlapping the final word, including a token that also contains trailing punctuation.

Therefore `tree bark,` activates `tree bark`, while `tree, bark` does not.
`camera-based` and `don't` each remain one word. Stage-1 counting uses the same
segmentation rule, so rebuilt inventories contain no cross-punctuation
bigrams or trigrams.

### 6.7 Collision handling

Several surface phrases can collapse to one canonical token or word tuple,
and finite hashes can share a bucket. The surface index stores complete canonical tuples and
exact surface identities, verifies every bucket candidate, and applies:

1. Exact boundary/alignment match.
2. Three source words over two, then one.
3. Higher corpus frequency for donor-row aliases.
4. Lower immutable row index as the stable final tie-break.
5. Every one-word sense retained when the dictionary route wins.

Surface-address collisions are written to `surface_index_collisions.jsonl`.
Cryptographic collisions receive a separate manifest count and are resolved by
complete-key comparison.

### 6.8 Longest 3→2→1 memory rule

At every complete-word endpoint:

```text
three-word donor Engram
otherwise two-word donor Engram
otherwise all one-word dictionary senses
```

The selected longest row supplies default memory injection. Lower-order rows
remain available for decomposition. Punctuation never gets a dictionary row;
its raw/canonical Qwen IDs continue through the live and Hash-Engram streams.

### 6.9 Surface-index and JTD acceptance tests

- Tokenizer revision, snapshot hashes, projection algorithm, projection fingerprint, and measured Qwen reduction appear in the manifest.
- Case, leading-space, accent, and whitespace-equivalent vocabulary pieces share canonical IDs.
- Comma, period, question, and other distinct punctuation classes retain distinct canonical keys.
- Every bank row produces canonical token and complete-word addresses.
- BOS/internal, variable token spans, 3→2→1 priority, and all-sense fallback pass.
- `tree bark,` matches even when punctuation is fused into the final token.
- `tree, bark` and cross-sentence pairs stay separated.
- Hash-Engram receives punctuation-bearing canonical suffixes on donor misses.
- Collision resolution is deterministic across repeated builds.
- Definition vectors pass through JTD unchanged.
- Layer-8, layer-24, and live maps fit toward same-content layer-2 anchors.
- Surface text and IDs remain metadata rather than vector channels.

### 6.10 Hash-Engram coverage on frozen donor misses

For a frozen donor miss, Dendritron hashes canonical local suffixes:

\[
z_{t,n,k}=\phi_{n,k}(x'_{t-n+1},\ldots,x'_t)
\]

A one-word dictionary match can coexist with those trainable rows. Thus the
exact route provides lexical/sense knowledge while Hash-Engram learns the
uncovered local token pattern, including punctuation structure.

---

## 7. Surface Lookup, JTD, LNGram, Skills, and Experts Have Different Jobs

| Mechanism | Address source | Role | First available |
|---|---|---|---|
| Surface exact lookup | Canonical projected Qwen IDs plus offset-aligned complete words | Retrieves rows from the completed frozen donor bank | Before reasoning round 1 |
| JTD latent transfer | Layer-8, layer-24, live states, and frozen layer-2 definition points | Places all active numerical views in the layer-2 concept frame | At sparse memory fusion |
| Hash Engram lookup | Deterministic hashes of canonical local Qwen-ID suffixes, including punctuation | Retrieves trainable local-pattern memory on frozen donor misses | Before reasoning round 1 |
| LNGram concept lookup | Discrete symbols learned from the current hidden state | Retrieves latent concept records and their skill priors | After an initial hidden state exists |
| Skill graph | Selected skill IDs | Returns the expert IDs already adjacent to each skill | After skill routing |
| Experience Engram lookup | LNGram concept, skill, and expert IDs | Retrieves prior successful coefficient traces, outcome records, and optional initialization priors | After concept and skill resolution |

Qwen token hashing and LNGram share the general idea of constant-time discrete
addressing. Their inputs and tables stay separate:

```text
raw Qwen IDs -> frozen P -> surface exact buckets / Hash Engram rows
live hidden states -> LNGram symbols -> latent concept rows
```

### 7.1 LNGram address construction

For hidden state \(h_t\):

\[
u_t=\operatorname{RMSNorm}(h_t), \qquad
z_t=u_tW_q
\]

Channels are divided into routes. Each route is hard-discretized into a symbol:

\[
a_{t,r}
=
\sum_{j=0}^{M-1}
\mathbf{1}[z_{t,(r,j)}>0]2^j
\]

An n-gram ending at \(t\) becomes an exact latent address:

\[
g_{t,r}^{(n)}
=
rK^n+
\sum_{i=0}^{n-1}
a_{t-n+1+i,r}K^i
\]

This is the LNGram mechanism retained from the attached paper: hidden state to discrete symbols to exact latent n-gram lookup.

### 7.2 Dendritron concept and adjacency record

LNGram supplies the latent address. Dendritron assigns the retrieved table value this graph-oriented schema:

```text
concept_record
  concept_id
  skill_prior[16..32]
  skill_ids[]
  skill_to_expert_edges[]
  experience_engram_ids[]
  success_statistics
```

The learned address can group related surface forms around a common latent concept. The current hidden state and retrieved donor Engrams supply the context needed to interpret that concept.

The geometric router evaluates the small bank of principal skill anchors. Once skill \(s\) is active, its adjacency record supplies:

\[
\mathcal E_s
=
\{e:\,(s,e)\in G_{\text{skill-expert}}\}
\]

An expert can have edges to several skills. Hundreds of expert nodes can therefore occupy one skill neighborhood while a composite expert can sit near several skills.

### 7.3 Dendritron readout adaptation

Retrieved definition and LNGram points use Euclidean locality in the joint
frame. For active point \(z_c\):

\[
d_c^2
=
\frac{1}{d}\left\|J_hh_t-z_c\right\|_2^2+\epsilon^2
\]

\[
a_c
=
\frac{d_c^{-n}}
{\epsilon+\sum_j d_j^{-n}}
\]

\[
m_t
=
J_{\mathrm{joint}\rightarrow h}
\left(
\sum_c a_cz_c-J_hh_t
\right)
\]

Every retrieved dictionary sense remains a distinct point with continuous
mass. Supported/opposing evidence and signed movement enter through Section
9's HarMax target-minus-distance derivative.

### 7.4 Search contract

Runtime routing follows this order:

1. LNGram retrieves one or more latent concept records.
2. Geometric routing scores the 16-32 principal skill anchors.
3. Each selected skill returns its stored expert adjacency.
4. The concept record and current context activate a bounded subset of those adjacent experts.
5. Each active expert instantiates its own branch activations in the current physical block.

The router's global geometric comparison remains skill-sized even when the committed expert library grows into the thousands.

---

## 8. Initial State and Staged Donor Injection

For recipient token IDs \(x_{1:T}\):

\[
H^{(0)}
=
E_{\text{tok}}(x_{1:T})
+
P_{\text{pos}}(1:T)
+
g_8P_8(E_{\text{JTD}}^{(8)})
\]

where:

- \(E_{\text{tok}}\) is the recipient token embedding.
- \(P_{\text{pos}}\) is fixed or learned positional geometry.
- \(P_8\) adapts the shallow phrase view before the first recurrent transition.
- Missing JTD rows contribute zero memory at initialization.

The deeper phrase view enters after the state has begun contextual processing:

\[
H_{\text{late}}'
=
H_{\text{late}}
+
g_{24}P_{24}(E_{\text{JTD}}^{(24)})
\]

Both gates begin near zero and are learned. The live state remains present
through residual addition. An initial recipient width of 2,048 keeps a direct
dimensional correspondence with both donor views. Width remains an explicit
experiment.

---

## 9. Geometric Attention

Geometric attention is a HarMax derivative field over a sparse contraction
pool. It acts directly on the current hidden state through distance geometry
and causally bound evidence.

For position \(i\), round \(r\), and physical block \(k\), construct:

\[
\mathcal P_i^{(r,k)}
=
\left\{
\left(
z_{iq},\omega_{iq},\sigma_{iq},\pi_{iq}
\right)
\right\}_{q=1}^{Q_i}
\]

where:

- \(z_{iq}\in\mathbb R^d\) is a projected hidden state, Engram, dictionary
  sense, LNGram concept, branch anchor, or branch-generated candidate;
- \(\omega_{iq}\geq 0\) is its evidence strength;
- \(\sigma_{iq}\in\{+1,-1\}\) marks supported versus opposing evidence;
- \(\pi_{iq}\) retains the exact source span, memory row, concept, expert, and
  branch pointer.

The pool is causally bounded. Hidden-state evidence comes from positions
\(j\leq i\), and a memory or concept anchor can enter only through a source
span ending at or before \(i\). Sparse JTD/LNGram retrieval, selected skill
adjacency, and active expert branches bound \(Q_i\); a global state-to-state
comparison is unnecessary.

Each physical block measures ordinary Euclidean pool distance with causal order
geometry:

\[
d_{iq}^2
=
\left\lVert h_i^{(r,k)}-z_{iq}\right\rVert_2^2
+
\lambda_p\delta_{iq}^2
+
\epsilon^2
\]

where \(\delta_{iq}\) is the relative causal displacement associated with
\(\pi_{iq}\). HarMax converts these distances into scale-invariant contraction
mass:

\[
p_{iq}
=
\frac{d_{iq}^{-n}}
{\sum_{\ell=1}^{Q_i}d_{i\ell}^{-n}}
\]

with harmonic exponent \(n>0\). Supported evidence defines the branch target
mass:

\[
y_{iq}
=
\frac{
\omega_{iq}\mathbf 1[\sigma_{iq}=+1]
}{
\sum_{\ell=1}^{Q_i}
\omega_{i\ell}\mathbf 1[\sigma_{i\ell}=+1]
}
\]

An executable pool contains at least one supported item with positive
\(\omega\). A branch whose required supported roles remain unbound stays
inactive for that visit and records the missing role in its trace.

Every opposing item remains in the HarMax denominator with \(y_{iq}=0\). The
harmonic residual is:

\[
\rho_i^{(r,k)}
=
-\sum_{q=1}^{Q_i}
y_{iq}\log p_{iq}
\]

The geometric-attention movement is the negative derivative of that residual:

\[
\boxed{
\Delta_{\mathrm{HarMax},i}^{(r,k)}
=
-\nabla_{h_i}\rho_i^{(r,k)}
=
n
\sum_{q=1}^{Q_i}
\left(y_{iq}-p_{iq}\right)
\frac{
z_{iq}-h_i^{(r,k)}
}{
d_{iq}^{2}
}
}
\]

The signed coefficient

\[
\gamma_{iq}=y_{iq}-p_{iq}
\]

has a direct meaning:

- \(\gamma_{iq}>0\): attraction toward underrepresented supported evidence;
- \(\gamma_{iq}<0\): repulsion from a competitor or overrepresented candidate;
- \(\gamma_{iq}=0\): local harmonic balance.

Thus attraction and repulsion arise from the HarMax derivative itself. The
residual \(\rho_i^{(r,k)}\) measures the pool's unresolved geometric mismatch
and supplies the confidence factor:

\[
c_i^{(r,k)}
=
\frac{1}{1+\rho_i^{(r,k)}}
\]

Each branch returns its derivative movement, residual, confidence, and exact
source pointers. Section 10.6 specifies how reasoning operators construct
their pools and how the soma combines the resulting signed movements.

---

## 10. Sparse Knowledge and the Universal/Shared-LoRA Subspace

### 10.1 Complete dictionary, sparse retrieval

The dictionary contains every retained English single-word headword from the
locked sources, including every distinct sense. Each sense row stores:

- exact source-grounded definition;
- ordered definition-word IDs;
- one frozen 2,048D BF16 Qwen layer-2 vector.

The completed Engram vocabulary is a mandatory coverage set inside this larger
dictionary. Full layer-2 rows remain in CPU/disk shards. Runtime selects senses
by surface lookup and fetches every matching row through
`FrozenDefinitionStore`. Each fetched row stays at its own frozen point while
joint-space locality controls its live influence. The dictionary therefore
supplies broad knowledge while live memory use remains proportional to active
word hits and their sense counts.

### 10.2 Phrase and definition payloads remain separate

The frozen phrase bank keeps its paired views:

```text
layer 8  -> early recipient assistance
layer 24 -> composed phrase meaning
```

The dictionary keeps layer 2 as the unchanged JTD reference frame. The
surface index preserves exact identities and returns the active phrase rows
or all matching sense rows. Separate \(J_8\), \(J_{24}\), and \(J_h\) maps
place phrase/live values into that frame; only active rows enter memory
fusion. The million phrase rows and the full dictionary bank stay outside the
recurrent state.

### 10.3 Universal/Shared-LoRA weight-space subspace

The reusable Universal/Shared-LoRA basis is built from successful task LoRA
factors. Its initial rank is 16 and can grow toward 32:

\[
U_{\text{skill}}
=
\operatorname{SVD/HOSVD}
\left(
\{\Delta W_{\text{successful tasks}}\}
\right)
\]

Dictionary, Engram, and LNGram addresses identify active knowledge. The
Universal/Shared-LoRA coordinates identify which learned operation should act
on that knowledge.

### 10.4 Fixed skill anchors and the working LoRA

Let \(r_s\in[16,32]\). For one recipient operator, store the principal factors:

\[
\beta\in\mathbb{R}^{d_{\text{out}}\times r_s},
\qquad
\alpha\in\mathbb{R}^{d_{\text{in}}\times r_s}
\]

Principal pair \(s\) is the stable skill anchor:

\[
S_s=\beta_{:,s}\alpha_{:,s}^{\top}
\]

For block visit \((r,k)\), geometric routing produces a sparse skill mask \(m^{(r,k)}\). The episode-local coefficients are:

\[
\epsilon_\beta^{(r,k)},
\epsilon_\alpha^{(r,k)}
\in\mathbb{R}^{r_s\times p}
\]

Mask the inactive skill rows:

\[
\widetilde\epsilon_\beta^{(r,k)}
=
\operatorname{diag}(m^{(r,k)})\epsilon_\beta^{(r,k)}
\]

\[
\widetilde\epsilon_\alpha^{(r,k)}
=
\operatorname{diag}(m^{(r,k)})\epsilon_\alpha^{(r,k)}
\]

The effective worked LoRA is:

\[
\Delta W_{\text{work}}^{(r,k)}
=
\left(\beta\widetilde\epsilon_\beta^{(r,k)}\right)
\left(\alpha\widetilde\epsilon_\alpha^{(r,k)}\right)^{\top}
\]

\[
\Delta h_{\text{skill}}^{(r,k)}
=
\Delta W_{\text{work}}^{(r,k)}h^{(r,k)}
\]

This gives the required separation:

- \(\alpha,\beta\): principal skill directions; fixed throughout the episode.
- \(m^{(r,k)}\): active skills; can change every block visit.
- \(\epsilon_{\alpha,\beta}^{(r,k)}\): current experience of those skills; can update every block visit in learning mode.
- \(\Delta W_{\text{work}}^{(r,k)}\): the effective LoRA used on the current state.

The attached Share paper uses this same central distinction: principal factors remain frozen during task adaptation while the lightweight coefficient state is trained. Dendritron applies that distinction inside its recurrent two-block episode.

### 10.5 Experts are knowledge/task/skill junctions

An expert records a reusable interaction between active knowledge, a requested
relation, useful principal skill directions, and branch wiring:

\[
e_j
\equiv
\left(
\mathcal K_j,
\mathcal C_j,
\mathcal T_j,
\mathcal S_j,
\mathcal B_j,
p_{\epsilon,j},
\text{success statistics}
\right)
\]

where:

- \(\mathcal K_j\) contains linked dictionary-sense and Engram IDs.
- \(\mathcal C_j\) contains linked LNGram concepts.
- \(\mathcal T_j\) identifies the task relation or requested outcome.
- \(\mathcal S_j\) is the sparse set of adjacent principal skill directions.
- \(\mathcal B_j\) is the persistent set of branch specifications.
- \(p_{\epsilon,j}\) is an optional reference to a successful coefficient
  prior in the experience tier.

The expert's identity is the cross-interaction junction. The prior can
initialize episode-local coefficients while remaining a separate record. One
expert can serve several skills; one skill can connect to many experts; a new
skill can initially have an empty expert neighborhood.

### 10.6 Expert-owned branches

Branches are persistent expert-owned reasoning paths. An expert stores the
specification while each physical-block visit instantiates fresh numerical
activations from the latest state.

For expert \(e\) and branch \(j\), the complete persistent specification is:

\[
\boxed{
\mathcal B_{e,j}
=
\left(
o_{e,j},
\mathcal R_{e,j},
\mathcal Q_{e,j},
m_{e,j},
\boldsymbol\sigma_{e,j},
\mathcal A_{e,j}
\right)
}
\]

where:

- \(o_{e,j}\) is the reasoning operator;
- \(\mathcal R_{e,j}\) is the relation the branch proposes or tests;
- \(\mathcal Q_{e,j}\) binds semantic source roles such as premise, cause,
  effect, candidate, intervention, or conclusion;
- \(m_{e,j}\) is the sparse mask over principal skill directions;
- \(\boldsymbol\sigma_{e,j}\) assigns supported/opposing signs to the bound
  roles;
- \(\mathcal A_{e,j}\) stores exact dictionary, Engram, LNGram, task, and
  relation anchors.

The shared branch-operator vocabulary is:

| Operator \(o_{e,j}\) | Pool construction | Branch action |
|---|---|---|
| **Deductive** | Bind all required premises, the proposed conclusion, and contradiction anchors. | Contract jointly supported premises and conclusion; repel conclusions that violate any bound premise. |
| **Causal** | Bind ordered cause, mechanism, and effect anchors with their causal span pointers. | Contract cause-to-effect-consistent states; repel reversed, temporally inconsistent, or mechanism-incompatible candidates. |
| **Contrastive** | Place supported candidates in positive roles and conflicting candidates in opposing roles. | Attract the supported distinction and repel alternatives according to their HarMax distance mass. |
| **Counterfactual** | Build a base pool and an intervention pool with one premise removed, replaced, or reversed. | Return the difference between the intervention and base derivative fields together with the change in harmonic residual. |
| **Abductive** | Bind an observed effect and locally retrieved candidate explanations. | Contract toward explanations that jointly account for the observation; repel candidates that leave a larger residual. |
| **Compositional/analogical** | Bind a source relation pair and target-domain anchors through shared skill directions. | Transfer the relation into the target pool and contract toward candidates preserving the same geometric relation. |

An expert stores only the branch specifications relevant to its
knowledge/task/skill junction. The operator vocabulary is shared, so thousands
of expert junctions can reuse the same small execution library.

For active expert \(e\), branch \(j\), block \(k\), and round \(r\), instantiate:

\[
\mathcal I_{e,j}^{(r,k)}
=
\operatorname{Bind}
\left(
\mathcal B_{e,j},
h^{(r,k)},
M_{\text{phrase}},
M_{\text{sense}},
M_{\text{LNGram}},
\epsilon_{\text{work}}^{(r,k)},
p_{\epsilon,e}
\right)
\]

Binding performs five operations:

1. Resolve every role in \(\mathcal Q_{e,j}\) to current hidden states,
   retrieved memories, concepts, or branch-generated candidates.
2. Project the bound values into the block's common hidden geometry while
   retaining every exact source pointer.
3. Apply the stored skill mask \(m_{e,j}\), relation
   \(\mathcal R_{e,j}\), and role signs \(\boldsymbol\sigma_{e,j}\).
4. Construct the causally valid HarMax pool
   \(\mathcal P_{e,j}^{(r,k)}\) and its supported target mass
   \(y_{e,j}^{(r,k)}\).
5. Evaluate Section 9's distances, HarMax mass, negative derivative, and
   harmonic residual.

The instantiated branch returns:

\[
b_{e,j}^{(r,k)}
=
\left(
\Delta_{e,j}^{(r,k)},
\rho_{e,j}^{(r,k)},
q_{e,j}^{(r,k)},
\Pi_{e,j}^{(r,k)}
\right)
\]

where:

- \(\Delta_{e,j}^{(r,k)}=-\nabla_h\rho_{e,j}^{(r,k)}\) is the signed HarMax
  movement;
- \(\rho_{e,j}^{(r,k)}\) is the unresolved harmonic residual;
- \(q_{e,j}^{(r,k)}\) is signed branch evidence;
- \(\Pi_{e,j}^{(r,k)}\) is the exact branch trace of sources, anchors, and
  intervention choices.

For ordinary branches:

\[
q_{e,j}^{(r,k)}
=
\frac{\varsigma_{e,j}^{(r,k)}}
{1+\rho_{e,j}^{(r,k)}}
\]

where \(\varsigma_{e,j}^{(r,k)}\in[-1,1]\) is the branch's verified relation
polarity. Let \(\widehat v_{\mathcal R,e,j}^{(r,k)}\) be the ordinary
Euclidean unit direction induced by the branch's bound source and conclusion
anchors. Then:

\[
\varsigma_{e,j}^{(r,k)}
=
\frac{
\left\langle
\Delta_{e,j}^{(r,k)},
\widehat v_{\mathcal R,e,j}^{(r,k)}
\right\rangle_2
}{
\epsilon+
\left\|\Delta_{e,j}^{(r,k)}\right\|_2
}
\]

This gives positive evidence when the derivative moves along the specified
relation and opposing evidence when it moves against that relation. For a
counterfactual branch:

\[
\Delta_{e,j}^{(r,k)}
=
\Delta_{\text{intervention}}^{(r,k)}
-
\Delta_{\text{base}}^{(r,k)}
\]

\[
\rho_{e,j}^{(r,k)}
=
\rho_{\text{intervention}}^{(r,k)}
-
\rho_{\text{base}}^{(r,k)}
\]

so the branch measures the consequence of the intervention rather than
repeating the base inference.

Its signed evidence is:

\[
q_{e,j}^{(r,k)}
=
\frac{
\rho_{\text{intervention}}^{(r,k)}
-
\rho_{\text{base}}^{(r,k)}
}{
1+
\left|
\rho_{\text{intervention}}^{(r,k)}
-
\rho_{\text{base}}^{(r,k)}
\right|
}
\]

A positive value means the intervention worsened harmonic consistency and the
altered premise was supporting the base relation. A negative value means the
intervention improved consistency and opposes the base relation.

The expert soma combines its active branches with signed L1 normalization:

\[
\boxed{
\Delta_e^{(r,k)}
=
\frac{
\sum_{j\in\mathcal J_e^{(r,k)}}
q_{e,j}^{(r,k)}
\Delta_{e,j}^{(r,k)}
}{
\epsilon+
\sum_{j\in\mathcal J_e^{(r,k)}}
\left|q_{e,j}^{(r,k)}\right|
}
}
\]

The block soma then combines the locally routed experts:

\[
\Delta_{\text{soma}}^{(r,k)}
=
\frac{
\sum_{e\in\mathcal E_{\text{active}}^{(r,k)}}
a_e^{(r,k)}\Delta_e^{(r,k)}
}{
\epsilon+
\sum_{e\in\mathcal E_{\text{active}}^{(r,k)}}
\left|a_e^{(r,k)}\right|
}
\]

where \(a_e^{(r,k)}\) is the local expert activation obtained from selected
skill adjacency, current LNGram concepts, and stored success evidence. Branch
specifications remain attached to their expert junctions; contraction pools,
movements, residuals, evidence, and traces are recreated on every visit.

### 10.7 Skill-first routing

For current state \(z\), geometric routing scores only the principal skill anchors:

\[
q_s
=
\widehat z^\top M_s\widehat S_s
\]

Select the active skill set by \(|q_s|\). Then retrieve the associated experts:

\[
\mathcal E_{\text{candidate}}
=
\bigcup_{s\in\mathcal S_{\text{active}}}
\mathcal E_s
\]

LNGram concept records and stored success metadata reduce this local set to the active experts. This is adjacency resolution around selected skills rather than a global geometric comparison against every expert node.

### 10.8 Episode learning, experience commit, and expert formation

During a learning-enabled episode, update the working coefficients with an explicit inner objective:

\[
\epsilon^{(r,k+1)}
=
\epsilon^{(r,k)}
-
\eta
\nabla_{\epsilon}
\mathcal L_{\text{episode}}^{(r,k)}
\]

The inner objective remains an open experimental choice. Candidate signals include observed-prefix prediction, answer-verification loss, geometric consistency, convergence energy, and direct outcome or user feedback.

At successful completion:

1. Retain the useful sparse coefficient trace and outcome evidence.
2. Append that experience record independently of expert identity.
3. Resolve the active knowledge anchor, concepts, task relation, and useful
   principal skills.
4. Reuse an existing expert junction when that cross-interaction is already
   represented.
5. Commit a new expert junction when the cross-interaction is novel and the
   outcome is verified.
6. Attach the successful coefficient record as an optional initialization
   prior and add LNGram concept-to-skill and skill-to-expert edges.
7. Reset the episode-local working coefficients for the next episode.

### 10.9 The x-part

For a learned update signal \(\delta\) and current orthonormal skill basis \(U\):

\[
c=U^\top\delta
\]

\[
x=(I-UU^\top)\delta
\]

- \(Uc\) uses existing skill directions.
- \(x\) records specialized update energy outside the current basis and remains
  attached to the successful coefficient/experience record.

Repeated, coherent x-parts indicate pressure for a new shared skill direction. Consolidation occurs when their energy and reuse exceed locked thresholds:

1. Merge the current basis and persistent x-directions.
2. Recompute a compact basis by SVD/HOSVD.
3. Keep rank within 16-32 unless a documented ablation supports expansion.
4. Reproject coefficient priors into the revised Shared-LoRA basis; Universal
   knowledge anchors remain in their separate geometry.
5. Freeze the revised basis for the next inference interval.

This separates three outcomes:

- Existing junction and useful prior: reuse the expert and its prior.
- New knowledge/task/skill cross-interaction: add an expert junction.
- Repeated structure outside the skill basis: add or replace a principal skill during consolidation.

---

## 11. Deep-Loop Reasoning

### 11.1 Two-block layerless recurrence

Store two ordered physical Dendritron blocks:

\[
\Phi=\left[\Phi_1,\Phi_2\right]
\]

Reuse them for \(R\) rounds:

\[
H^{(r,1)}
=
\Phi_1
\left(
H^{(r,0)},
M_{\text{JTD}},
M_{\text{LNGram}},
\epsilon^{(r,0)}
\right)
\]

\[
H^{(r+1,0)}
=
\Phi_2
\left(
H^{(r,1)},
M_{\text{JTD}},
M_{\text{LNGram}},
\epsilon^{(r,1)}
\right)
\]

where \(H^{(0,0)}=H^{(0)}\). Thus:

\[
\text{stored depth}=2,
\qquad
\text{effective thought depth}=2R
\]

Each physical-block visit performs:

1. LNGram concept-address update
2. JTD/LNGram memory resolution
3. Geometric skill selection
4. Skill-to-expert adjacency resolution
5. Expert-owned branch-pool construction
6. HarMax geometric/memory/branch contraction
7. First DeepLoop residual update and RMSNorm
8. Working LoRA and expert-soma computation
9. Second DeepLoop residual update and RMSNorm
10. Working-coefficient update in learning mode
11. Stop/convergence measurement

### 11.2 Recurrent update

Each physical block contains two sequential residual sublayers. Let
\(H_{\mathrm{in}}^{(r,k)}\) be the state entering physical block \(k\) during
round \(r\).

The first sublayer executes HarMax geometric interaction, sparse memory
injection, and the active reasoning-branch contraction pools:

\[
C_k^{(r)}
=
\Delta_{\text{HarMax-context}}^{(r,k)}
+
\Delta_{\text{memory}}^{(r,k)}
+
\Delta_{\text{branch-contract}}^{(r,k)}
\]

\[
\boxed{
U^{(r,k)}
=
\operatorname{RMSNorm}
\left(
\alpha H_{\mathrm{in}}^{(r,k)}
+
C_k^{(r)}
\right)
}
\]

The second sublayer applies the selected Shared-LoRA skills and the expert
somas produced from those branch activations:

\[
E_k^{(r)}
=
\Delta_{\text{work-LoRA}}^{(r,k)}
+
\Delta_{\text{expert-soma}}^{(r,k)}
\]

\[
\boxed{
H_{\mathrm{out}}^{(r,k)}
=
\operatorname{RMSNorm}
\left(
\alpha U^{(r,k)}
+
E_k^{(r)}
\right)
}
\]

The state flows sequentially:

\[
H_{\mathrm{in}}^{(r,k)}
\longrightarrow
U^{(r,k)}
\longrightarrow
H_{\mathrm{out}}^{(r,k)}
\]

For Block 1,
\(H_{\mathrm{in}}^{(r,1)}=H^{(r,0)}\) and
\(H^{(r,1)}=H_{\mathrm{out}}^{(r,1)}\). For Block 2,
\(H_{\mathrm{in}}^{(r,2)}=H^{(r,1)}\) and
\(H^{(r+1,0)}=H_{\mathrm{out}}^{(r,2)}\).

The skip scale \(\alpha\) is present in both forward updates. The DeepLoop
gain \(\beta\) initializes the designated matrices inside both residual
branches and is applied once at initialization.

### 11.3 Two-block complementary bias

The blocks share the same operator vocabulary and have different recurrent
jobs:

| Physical block | Bias | Pool and branch behavior | Result carried forward |
|---|---|---|---|
| **Block 1: expansion** | Retrieve, associate, and propose. | Preserve several plausible anchors; emphasize abductive, compositional/analogical, causal-proposal, and counterfactual-generation paths; attach a HarMax residual to every proposal. | A richer candidate field with exact memory, concept, expert, branch, and source pointers. |
| **Block 2: contraction** | Contrast, verify, and integrate. | Re-evaluate Block 1 proposals with deductive, causal-consistency, contrastive, and counterfactual tests; use opposing anchors as repulsive mass; weight branch movement by residual confidence before the soma. | A concentrated hidden state whose surviving relations have lower supported residual and explicit rejected alternatives. |

The bias is implemented by block-specific pool gates, role-binding priors, and
operator priors inside \(\Psi_1\) and \(\Psi_2\). Both blocks retain every
operator in Section 10.6, allowing later rounds to revise an earlier proposal
or reopen a prematurely contracted result.

One complete reasoning round is:

\[
H^{(r,0)}
\xrightarrow[\text{expand}]{\Phi_1}
H^{(r,1)}
\xrightarrow[\text{contract}]{\Phi_2}
H^{(r+1,0)}
\]

The output of Block 2 returns to Block 1, producing the recurrent pattern:

\[
\text{proposal}
\rightarrow
\text{verification}
\rightarrow
\text{revised proposal}
\rightarrow
\text{stronger verification}
\]

Expansion and contraction describe candidate/evidence geometry. RMSNorm and
DeepLoop govern hidden-state magnitude throughout both roles.

### 11.4 DeepLoop stability rule

DeepLoop gives the tied-visit stability condition:

\[
M\kappa_R
\left(\frac{\beta}{\alpha}\right)^2
=O(1)
\]

where:

- \(M\) is the number of residual-branch visits across all rounds.
- \(\kappa_R\) measures alignment of updates and sensitivities across repeated visits.
- \(K=2\) is the number of stored physical blocks.
- \(N=KR=2R\) is block-equivalent unrolled depth.
- Every block has two residual sublayers, so \(M=2KR=2N=4R\).

Dendritron uses the aligned tied-loop exponent \(p=1/2\):

\[
\boxed{
\alpha=(2N)^{1/2},
\qquad
\beta=(8N)^{-1/2}
}
\]

For \(K=2\):

\[
\alpha=\sqrt{4R}=2\sqrt R,
\qquad
\beta=\frac{1}{\sqrt{16R}}=\frac{1}{4\sqrt R}
\]

Every residual sublayer applies:

\[
x_{v+1}
=
\operatorname{RMSNorm}
\left(
\alpha x_v+f_j(x_v;\phi_j)
\right),
\qquad
v=1,\ldots,4R
\]

The same physical parameters \(\phi_j\) are reused at their corresponding
sublayer on every round. Visit alignment \(\kappa_R\) remains a required
training diagnostic; \(\alpha\), \(\beta\), and Post-RMSNorm remain fixed by
the DeepLoop rule.

### 11.5 Adaptive reasoning

Training begins with fixed unroll counts such as \(R\in\{1,2,4,8\}\). Adaptive stopping activates after the core is stable:

\[
\rho_r
=
\frac{\|H^{(r+1)}-H^{(r)}\|}
{\epsilon+\|H^{(r)}\|}
\]

After a minimum round count, execution can stop when \(\rho_r\) remains below a threshold and the output token ranking is stable.

---

## 12. Softmax-Free Vocabulary Output

Tie or align the output vocabulary vectors with the recipient token embedding table:

\[
\ell_v
=
\widehat{W_oh_t}^{\top}
G
\widehat{E_{\text{tok}}[v]}
\]

The model emits raw geometric scores \(\ell_v\).

Initial training objective:

\[
\mathcal{L}_{\text{rank}}
=
\sum_{v^-\in\mathcal{N}}
\max
\left(
0,
\gamma-\ell_{v^+}+\ell_{v^-}
\right)
\]

where \(v^+\) is the true next token and \(\mathcal{N}\) contains sampled hard negatives.

Initial decoding experiments:

- Raw-score argmax
- Top-k raw-score selection
- Rank-based stochastic sampling

Evaluation emphasizes next-token rank, MRR, top-k accuracy, sequence exact match, downstream task accuracy, and generation quality.

---

## 13. Runtime Sequence

For each input:

1. **Tokenize once:** text \(\rightarrow\) raw Qwen IDs and fast character offsets.
2. **Project memory addresses:** apply frozen \(P\) to a parallel ID copy;
   raw IDs continue to embeddings and the output vocabulary.
3. **Align complete words:** map Unicode words onto Qwen token positions;
   punctuation/symbol material between words resets the frozen n-gram segment.
4. **Longest resolution:** at every complete-word endpoint select exact
   trigram, otherwise bigram, otherwise all one-word dictionary senses.
5. **Hash-Engram miss coverage:** a position without a frozen trigram/bigram
   hit hashes canonical local Qwen suffixes, retaining punctuation classes.
6. **Retrieve frozen values:** selected donor row \(\rightarrow\) paired
   layer-8/layer-24 Engrams; one-word fallback \(\rightarrow\) layer-2 senses.
7. **Dictionary grounding:** retain ordered definition-word links as metadata
   and fetch every exact sense row for the active word.
8. **Joint transfer:** keep layer-2 senses fixed, map active layer-8/layer-24
   phrase rows and the live state into their shared concept frame, and compute
   continuous locality mass over the retrieved points.
9. **Initialize state:** token and position geometry plus near-zero-gated
   layer-8 phrase memory, dictionary knowledge, and active Hash-Engram values.
10. **LNGram discretize:** current state \(\rightarrow\) latent concept addresses and concept records.
11. **Skill routing:** compare the current state against the 16-32 principal skill anchors.
12. **Expert adjacency:** selected skills and LNGram concepts \(\rightarrow\) adjacent expert IDs.
13. **Branch binding:** instantiate each selected
    \(\mathcal B_{e,j}\), bind its semantic roles, preserve its exact pointers,
    and construct its causal contraction pool.
14. **First residual sublayer:** compute HarMax distance mass, signed
    attraction/repulsion, harmonic residuals, memory movement, and raw branch
    contraction; apply \(\operatorname{RMSNorm}(\alpha H+C)\).
15. **Second residual sublayer:** apply the working Shared-LoRA skills,
    combine branch movements through expert and block somas, and apply
    \(\operatorname{RMSNorm}(\alpha U+E)\).
16. **Block 1 expansion:** retain several relation and explanation proposals
    with their branch residuals and traces.
17. **Block 2 contraction:** test the proposals with opposing evidence,
    deduction, causal consistency, contrast, and counterfactual intervention;
    integrate the supported result.
18. **Deep loop:** return Block 2's hidden state to Block 1 and repeat for the
    round budget or convergence rule.
19. **Working-coefficient update:** update episode-local skill coefficients in
    learning mode.
20. **Commit:** after verified success, append the coefficient/outcome
    experience, then reuse or append the corresponding expert junction and
    update graph edges.
21. **Decode:** final state \(\rightarrow\) raw Qwen-vocabulary token score.

---

## 14. Build Order

### Stage 3S - Acquire and measure canonical definitions

Implemented deliverables:

- `stage3_dictionary/definition_sources.py`
- `stage3_dictionary/prepare_definition_sources.py`
- `modal_prepare_definition_sources.py`
- `DEFINITION_SOURCES.md`
- four canonical versioned JSONL assets
- immutable definition-source manifest
- Engram-vocabulary coverage report
- exact dictionary storage report before Qwen extraction

This stage compiles the complete curated single-word dictionary and every
retained sense. Engram-word coverage must reach 100%. This is CPU/network work
and completes before Qwen donor inference.

### Stage 3A - Build dictionary inventory and word graph

Implemented deliverables:

- `stage3_dictionary/build_dictionary_inventory.py`
- `words.jsonl`
- `senses.jsonl`
- `dictionary.sqlite3`
- ordered definition-word edges

### Stage 3B - Extract dictionary layer-2 knowledge

Implemented pipeline:

- `modal_extract_definition_states.py`
- one BF16 `layer02` row per sense
- resumable 50k-row shards
- sense lookup database and manifest

### Stage 3C - Optional definition-vector diagnostic

`stage4_subspace/build_universal_subspace.py` remains available as a CPU-only
PCA diagnostic. Its output is excluded from the Dendritron runtime and required
build path. The 16-direction/98%-variance test belongs to successful
task-adapter factors in Stage 8.

### Stage 3D - Implement LNGram

Implemented module:

- `dendritron/lngram.py`
- 4-bit routes
- exact 2/3-gram route-partitioned tables
- one-bit counterfactual routing gradient
- signed geometric readout

### Stage 3E - Implement longest memory routing and expert topology

Implemented modules:

- `dendritron/retrieval.py`
- exact word-order 3→2→1 resolution over variable recipient-token spans
- one-word sense preservation and lower-order decomposition access
- `dendritron/expert_graph.py`
- many-to-many knowledge/task/skill junctions
- optional coefficient-prior references separate from expert identity

### Stage 4A - Lock Qwen and build the canonical memory projection

Implemented deliverable:

`tokenizer_contract.json`

`canonical_token_projection.json`

Required fields:

- model or tokenizer identifier
- exact revision
- vocabulary size
- special-token IDs
- tokenizer file hashes
- BOS and internal-space examples
- exact Stage-2 resolved revision
- `add_special_tokens=False`
- official Engram normalization algorithm identifier
- full raw-to-canonical integer table and fingerprint
- raw/effective Qwen vocabulary sizes and measured reduction
- 23.43% paper result recorded only as the 128k reference measurement

The first recipient uses the Qwen donor tokenizer. Its resolved revision comes
from the completed Stage-2 manifest. Raw IDs remain exact for modeling; the
canonical table is frozen and serves memory addresses only.

### Stage 4B - Build the surface-memory index

Implemented code:

- `stage3_jtd/build_joint_transfer_domain.py`
- `modal_build_joint_transfer_domain.py`
- `dendritron/jtd.py`
- `dendritron/engram_store.py`
- `dendritron/definition_store.py`
- `dendritron/hash_engram.py`
- `dendritron/memory_pipeline.py`
- `surface_index.sqlite3`
- `surface_index_collisions.jsonl`
- `surface_index_manifest.json`

The surface database embeds the canonical projection, fast token-tuple addresses,
and complete-word addresses while retaining `bank_name`,
`word_order`, and independent row indices. The layer-8/layer-24 values remain
in the original separate bigram and trigram shards. Layer-2 dictionary values
remain in CPU/disk shards and are fetched for every exact matching sense.

CPU data command:

```bash
modal run modal_build_joint_transfer_domain.py \
  --stage2-root /data/dendritron-stage2-punctuation-v2
```

Matching completed artifacts resume from their manifest. A changed Stage-2
manifest, key file, dictionary database, or tokenizer revision stops before
replacement.

### Stage 4C - Fit the Joint Transfer Domain

Implemented code:

- `dendritron/joint_transfer.py`
- `stage3_jtd/fit_joint_transfer_domain.py`
- `modal_extract_jtd_latent_assets.py`
- `modal_fit_joint_transfer_domain.py`

The layer-2 definition vectors define the fixed reference geometry. An offline
Qwen job exports its complete input-embedding matrix and deterministic phrase
anchors containing same-text layer-2, layer-8, and layer-24 views. The CPU
fitter learns the layer-8 and layer-24 maps with point alignment and
neighborhood preservation. The live 2,048D recipient map starts as identity
and trains with the recipient unless genuine paired live/reference states are
provided. Surface strings and IDs remain outside the numerical archive. The
fitted checkpoint loads through `DendritronLM.load_joint_transfer_checkpoint`.

The anchor job runs only after the punctuation-v2 inventory audit has selected
the canonical Stage-2 rows:

```bash
modal run modal_extract_jtd_latent_assets.py --smoke \
  --stage2-root /data/dendritron-stage2-punctuation-v2
modal run modal_extract_jtd_latent_assets.py \
  --stage2-root /data/dendritron-stage2-punctuation-v2
modal run modal_fit_joint_transfer_domain.py
```

### Stage 4D - Measure sparse recipient projections

Deliverables:

- `analyze_engram_subspace.py`
- layer-8 spectrum
- layer-24 spectrum
- cross-view spectrum
- rank comparison at 8/16/24/32/64/128/256
- recipient projection candidates for selected layer-8, layer-24, and layer-2
  payloads
- word-sense and phrase-to-definition routing report

### Stage 5 - Build a minimal recurrent core

Include:

- token embeddings
- position geometry
- surface-memory lookup
- layer-8/layer-24/live JTD projection
- HarMax derivative contraction pools
- signed attraction/repulsion and harmonic residual reporting
- two physical blocks, each with two sequential residual sublayers
- tied vocabulary geometry

Acceptance test: overfit a tiny corpus and confirm memory hits change token rank in the expected direction.

### Stage 6 - Integrate LNGram with the recurrent core

Include:

- connect the implemented binary latent symbols to live recurrent states
- train route-partitioned exact lookup values
- measure the implemented counterfactual surrogate gradient
- compare signed geometric readout with the paper's sigmoid readout

Acceptance test: latent-key consistency, table utilization, and performance on JTD misses.

### Stage 7 - Add skills, experts, and branches

Include:

- rank-16 principal skill factors
- episode-local working LoRA coefficients
- skill-first geometric router
- LNGram/skill-to-expert adjacency
- fixed expert knowledge/task/skill junction records
- complete \(\mathcal B_{e,j}\) specifications
- deductive, causal, contrastive, counterfactual, abductive, and
  compositional/analogical operators
- per-visit HarMax pool construction and branch activations
- branch/soma execution in both physical blocks
- Block 1 expansion and Block 2 contraction biases
- expert utilization ledger

### Stage 8 - Activate deep-loop training

Include:

- fixed \(R=1,2,4,8\) ablation
- two stored physical blocks for every \(R\)
- two residual sublayers per physical block
- fixed aligned-loop scaling
  \(\alpha=(2N)^{1/2}\), \(\beta=(8N)^{-1/2}\)
- \(\kappa_R\) measurement
- adaptive stop after stable fixed-depth training

### Stage 9 - Activate coefficient experience, expert formation, and continual x-part learning

Include:

- episode inner-objective ablation
- worked-coefficient update and reset
- knowledge/task/skill junction novelty threshold
- optional coefficient-prior attachment
- experience Engram append
- LNGram concept-to-skill and skill-to-expert edge updates
- orthogonal residual measurement
- temporary x-directions
- consolidation thresholds
- coordinate reprojection tests
- forgetting and backward-transfer tests

---

## 15. Required Ablations

| ID | Configuration | Question |
|---|---|---|
| A0 | Geometric recurrent core | Can the core learn token transitions? |
| A1 | A0 + JTD frozen memory | Does exact grafted knowledge improve rank and reasoning? |
| A2 | A1 + LNGram | Does latent lookup improve misses and emerging patterns? |
| A3 | A2 + shared rank-16 skill basis | Do reusable directions improve task transfer? |
| A4 | A3 + skill adjacency + fixed experts + expert branches | Does skill-local expert execution improve efficiency or quality? |
| A5 | A4 + two physical blocks at \(R=2,4,8\) | Does recurrent depth improve compositional reasoning? |
| A6 | A5 + coefficient experience and expert-junction commit | Do successful traces become reusable priors while expert identity remains stable across related tasks? |
| A7 | A6 + x-part consolidation | Does continual growth add skills while preserving prior episodes? |

All comparisons record active parameters, stored parameters, FLOPs, wall time, peak memory, token rank metrics, retrieval hit rate, and reasoning-task performance.

---

## 16. MERIDIAN Ledger

Every training run records:

- JTD bigram/trigram hit rate
- JTD collision count and resolution rate
- LNGram route utilization and key entropy
- Layer-8 versus layer-24 contribution
- HarMax contraction-pool size and source composition
- Harmonic exponent and distance distribution
- Attraction versus repulsion mass from \(y-p\)
- Branch harmonic residual and residual-confidence distribution
- Block 1 proposal count and Block 2 survivor count
- Skill selection frequency and neighborhood size
- Expert selection frequency
- Expert skill-edge and concept-edge sparsity
- Working-coefficient delta per block visit
- Branch contribution norms
- New expert-junction commit, reuse, and merge counts
- Coefficient-prior reuse counts
- Experience Engram append count
- Concept-to-skill and skill-to-expert edge growth
- Loop delta \(\rho_r\)
- Visit alignment \(\kappa_R\)
- Skill-basis explained update energy
- x-part residual energy
- Consolidation events and coordinate reprojection error
- Next-token rank, MRR, and top-k accuracy
- Task accuracy, compute, latency, and memory

This ledger separates architectural claims from impressions.

---

## 17. Open Decisions

| ID | Decision | Fastest valid test |
|---|---|---|
| O-001 | Later custom-tokenizer ablation after the locked raw-Qwen plus canonical-memory baseline | Compare sequence length, output-head cost, memory hit rate, and CPU JTD rebuild time against Qwen. |
| O-002 | Recipient state width | Compare 1,024 and 2,048 while keeping memory projections compute-matched. |
| O-003 | Initial control rank | Compare 16, 24, and 32 using spectral and task-routing metrics. |
| O-004 | Active experts per token/round | Compare 1, 2, and 4. |
| O-005 | LNGram bit width and route count | Measure utilization, collisions, and JTD-miss performance. |
| O-006 | Memory-view fusion | Compare layer 8, layer 24, additive learned fusion, and separate branches. |
| O-007 | Fixed and adaptive loop schedule | Compare \(R=1,2,4,8\), then calibrate the convergence threshold under the locked \(p=1/2\) DeepLoop scaling. |
| O-008 | Decoding policy | Compare argmax, top-k raw scores, and rank-based stochastic sampling. |
| O-009 | Episode inner objective | Compare prefix prediction, verifier feedback, geometric consistency, and outcome feedback. |
| O-010 | Expert branch count and micro-operator | Compare one, two, and four branches per active expert with compute matching. |
| O-011 | Expert commit threshold | Calibrate knowledge/task/skill junction novelty, success confidence, and expert reuse. |
| O-012 | Maximum local experts per skill visit | Compare bounded adjacency retrieval at 8, 32, and 128 candidates. |

---

## 18. Recommended Repository Layout

```text
dendritron
  /specs
    DENDRITRON_MASTER_SPEC.md
    DECISION_LOG.md
  /manifests
    corpus_manifest.json
    donor_bank_manifest.json
    tokenizer_contract.json
    canonical_token_projection.json
    surface_index_manifest.json
  /stage1_corpus
    corpus_builder.py
  /stage2_grafting
    modal_extract_states.py
  /stage3_jtd
    build_joint_transfer_domain.py
  /stage4_subspace
    analyze_engram_subspace.py
  /dendritron
    tokenizer.py
    jtd.py
    retrieval.py
    engram_store.py
    hash_engram.py
    memory_pipeline.py
    lngram.py
    geometric_attention.py
    skill_basis.py
    skill_graph.py
    expert_graph.py
    experience_engram.py
    working_adapter.py
    recurrent_core.py
    output_geometry.py
    meridian.py
  /tests
    test_jtd_roundtrip.py
    test_retrieval.py
    test_expert_graph.py
    test_memory_rows.py
    test_harmax_contraction.py
    test_branch_operators.py
    test_loop_stability.py
    test_skill_consolidation.py
```

Existing files can remain in place until the next implementation step; this layout is the target organization.

---

## 19. Source Map

The architecture draws specific mechanisms from the attached project sources:

| Source | Mechanism retained |
|---|---|
| `Memory Grafting Scaling Language Model.pdf`, Sections 3.1-3.2 and Appendix A.2 | Exact phrase identity, donor-tokenizer offline value construction, recipient-tokenizer online key, final-token state, longest suffix match, and trainable hash-based Engram fallback |
| `Conditional Memory via Scalable Lookup.pdf`, Section 2.2 and Appendix C, plus the official Engram demonstration | Surjective canonical token projection; NFKC/NFD, accent, case, and whitespace algorithm; 23.43% reference reduction on the paper's 128k tokenizer; deterministic n-gram retrieval |
| `Lngram.pdf`, Sections 2.2-2.4 and 4.1 | Hidden-state discretization, multi-route latent symbols, exact latent n-gram lookup, and progressively formed latent concepts |
| `Shared LoRA Subspaces for almost Strict Continual Learning.pdf`, Section 3.2 | Principal factors frozen during adaptation, trainable lightweight coefficients, temporary factors, SVD consolidation, and analytical coefficient reprojection |
| `Harmonic Loss Trains Interpretable AI Models`, Section 2, arXiv:2502.01628v2 | Euclidean-distance HarMax mass, scale invariance, finite centers, and the harmonic cross-entropy whose derivative supplies Dendritron's contraction field |
| `Depth Scaling for Looped Transformers.pdf`, Section 3 | Two residual sublayers per physical block, Post-RMSNorm tied visits, visit alignment, and the \(p=1/2\) loop-aware \(\alpha,\beta\) scaling |

The Dendritron-specific synthesis is the combination of:

- Frozen full-state Engrams
- Raw-Qwen/canonical-memory dual ID streams
- Offset-aligned complete-word surface lookup with punctuation barriers
- Trainable canonical-Qwen-token Hash Engram with punctuation retained
- LNGram latent lookup
- HarMax derivative contraction pools
- Signed attraction/repulsion and harmonic residual routing
- Principal skill anchors
- Episode-local worked LoRA coefficients
- Skill-to-expert adjacency
- Knowledge/task/skill expert junctions
- Optional coefficient priors stored separately from expert identity
- Expert-owned reasoning-operator branches and somas
- Appendable experience Engrams
- Two physical blocks reused recurrently
- Block 1 expansion and Block 2 contraction
- Two sequential residual sublayers per physical block
- DeepLoop stability
- Softmax-free vocabulary geometry

---

## 20. Immediate Next Move

The next data-producing run is:

> Recount the corpus into a separate punctuation-v2 root, then compare its
> top-500k bigram and trigram keys against the completed Stage-2 keys. This
> establishes exactly which donor rows survive unchanged and which corrected
> phrases require extraction.

```bash
modal run corpus_builder.py \
  --output-root /data/dendritron-stage1-punctuation-v2 \
  --force
modal run modal_compare_punctuation_inventory.py
```

Review
`/data/dendritron-stage1-punctuation-v2/audit/punctuation_inventory_audit.json`.
Preserve every exact UTF-8 overlap row from Stage 2, extract donor states only
for added corrected rows, and write a new canonical Stage-2 root. Then:

1. Compile the corrected CPU surface index.
2. Export the complete Qwen input-embedding table and real JTD anchor pairs.
3. Fit layer-8 and layer-24 maps on CPU into the unchanged layer-2 definition
   frame.
4. Load the symbol table, dictionary bank, phrase rows, and transfer maps into
   the CPU/ARM recipient.
5. Train the recipient and its identity-initialized live-to-joint map.
