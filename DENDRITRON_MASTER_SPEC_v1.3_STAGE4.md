# DENDRITRON
## Master Build Specification and Project Ledger

Version: 1.4  
Date: 2026-08-15  
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

> **v1.4 live-state, LNGram, and composed-skill lock:** frozen donor rows are
> conditional-memory values, while the recipient's evolving DeepLoop state is
> the working representation that performs the current task. LNGram creates
> discrete addresses from that live state and retrieval enters it as a gated
> residual. Procedural SVD/HOSVD proposes and labels skill slots from successful
> transitions. Every populated skill owns a complete block-specific private
> LoRA plus coefficients over the complete frozen shared basis. Runtime keeps
> persistent weights fixed; verified offline replay may update the selected
> skill's private factors, and controlled consolidation may revise the shared
> basis.

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

Dendritron uses the exact Qwen donor tokenizer, addresses frozen layer-2
definition and layer-8/layer-24 phrase memory through exact and LNGram routes,
injects retrieved evidence as a gated residual into a distinct live DeepLoop
state, and repeatedly transforms that state with routed composed skill LoRAs,
expert-owned branches, and exact typed registers through two stored HarMax
blocks; successful procedural trajectories supply SVD-labeled skill slots and
verified offline updates.

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
| D-010 | **REVOKED:** LNGram addresses connect concept records, skill nodes, expert adjacency, and experiential memory. Replaced by D-077 through D-079: LNGram addresses word-definition memory only. | **REVOKED** |
| D-011 | **REVOKED:** The Shared-LoRA skill subspace begins with 16 principal factor directions and can grow toward 32 when persistent residual evidence supports consolidation. Replaced by D-047 through D-052. | **REVOKED** |
| D-012 | **REVOKED:** Principal Shared-LoRA factors are themselves the complete skill directions, and episode-local coefficients alone learn the current task. Replaced by the composed per-skill adapter in D-048 through D-052. | **REVOKED** |
| D-013 | **REVOKED:** An expert directly links LNGram concepts with task, skill, and branch records. Replaced by D-078. | **REVOKED** |
| D-014 | **REVOKED:** Episode-local LoRA coefficients update at every block visit in learning mode. Replaced by frozen runtime parameters and verified offline private-skill updates in D-052. | **REVOKED** |
| D-015 | DeepLoop's aligned tied-loop rule governs recurrent stability: every residual-sublayer visit uses Post-RMSNorm with \(\alpha=(2N)^{1/2}\), while designated residual matrices receive initialization gain \(\beta=(8N)^{-1/2}\). Visit alignment \(\kappa_R\) is measured as a diagnostic. | **LOCKED** |
| D-016 | **REVOKED:** Selected skills reach experts through an LNGram/skill graph. Replaced by D-079. | **REVOKED** |
| D-017 | Branch structure belongs to the expert junction. Its branch specification persists with the expert, while numerical branch activations are recreated from the current live state, router gates, and typed registers at every physical-block visit. | **LOCKED** |
| D-018 | Frozen donor Engrams and learned experience Engrams occupy separate tiers. Successful trajectory traces, routed skills, offline update evidence, and outcome records append to the experience tier while the million donor rows remain immutable. | **LOCKED** |
| D-019 | The dictionary bank stores one immutable row per word sense. Every sense preserves its exact definition, ordered definition-word IDs, and one 2,048D Qwen hidden-state vector from `hidden_states[2]` at the final token of a fixed readout marker. | **LOCKED** |
| D-020 | The dictionary uses a shallow layer-2 donor view so the record preserves lexical/compositional definition structure. Layer-24 dictionary summaries are excluded from the canonical bank. | **LOCKED** |
| D-021 | The dictionary is a complete curated single-word sense bank. Every retained word sense has an exact definition, ordered definition-word IDs, and one frozen layer-2 row. The bank remains on CPU/disk and runtime fetches selected rows sparsely. | **LOCKED** |
| D-022 | **REVOKED:** Dendritron has one fixed 16-32D skill subspace whose individual directions are the complete skills. Replaced by the shared/private composed-skill contract in D-046 through D-052. Dictionary vectors remain an independently addressed knowledge bank. | **REVOKED** |
| D-023 | The JTD reference frame is the frozen Qwen layer-2 definition geometry. Definition rows pass through unchanged; separate learned maps align layer-8 Engrams, layer-24 Engrams, and live recipient states to that frame. | **LOCKED** |
| D-024 | Runtime memory resolution uses the longest exact word-order match ending at each position: three-word donor Engram, otherwise two-word donor Engram, otherwise all one-word dictionary sense points. Context interacts with those points continuously in the joint geometry. | **LOCKED** |
| D-025 | **REVOKED:** A universal 16-direction/98%-variance target governs the procedural basis. Replaced by measured energy thresholds and separate semantic/procedural SVDs in D-046, D-047, and D-054. | **REVOKED** |
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
| D-043 | Frozen donor rows and the live recipient state are different objects. \(m^{(2)},m^{(8)},m^{(24)}\) denote immutable memory values; \(h_t\) denotes the evolving 2,048D DeepLoop working state. | **LOCKED** |
| D-044 | Every memory read enters \(h_t\) through a learned or calibrated gated residual. The live state persists across recurrent visits and remains the carrier of the current problem, intermediate reasoning, and task context. | **LOCKED** |
| D-045 | **REVOKED:** LNGram addresses resolve donor/definition memory plus general graph adjacency. Replaced by D-077: their runtime payload is confined to word-definition memory and definition adjacency. | **REVOKED** |
| D-046 | Procedural SVD/HOSVD uses successful live-state transitions, operator updates, adapter updates, or Jacobian sketches. Semantic-memory SVD remains a separate diagnostic over Engram/definition rows. | **LOCKED** |
| D-047 | A retained procedural principal direction proposes and labels a skill slot; it is an address or dominant fingerprint rather than a restriction on the skill's input. Every skill receives the complete live state and therefore all retained task variance. | **LOCKED** |
| D-048 | For skill \(s\) and block \(b\), the adapter is \(\Delta W_{s,b}=B_b^{sh}\operatorname{diag}(q_s)A_b^{sh}+B_{s,b}^{priv}A_{s,b}^{priv}\). The shared and private terms form one complete skill LoRA. | **LOCKED** |
| D-049 | \(q_s\) spans the entire shared basis and is shared across blocks to preserve skill identity. \(A_b^{sh},B_b^{sh}\) are block-specific so expansion and contraction can realize the shared procedure differently. Dense learned \(q_s\) vectors are allowed. | **LOCKED** |
| D-050 | Every populated skill owns block-specific private factors. \(A_{s,b}^{priv}\) reads every input dimension and \(B_{s,b}^{priv}\) can affect every output dimension; private rank is a bottleneck capacity chosen from residual SVD energy. Rank 4 is provisional. | **LOCKED** |
| D-051 | The implementation exposes a capacity bank plus populated/active masks. The current 21 slots are placeholder capacity rather than an ontology or mathematical limit. If measured task data retains 32 procedural modes, as many as 32 skill slots may be populated while every skill still uses the complete shared basis and its complete private adapter. | **LOCKED** |
| D-052 | During inference and task execution, core, shared basis, coefficients, experts, and private skill weights are frozen. After verified success, the selected skill's private factors may train in an offline working buffer, pass replay and regression validation, then commit as a versioned update. Repeated reusable residual structure may enter a controlled shared-basis consolidation. | **LOCKED** |
| D-053 | Mathematical and symbolic work combines the continuous live semantic state with exact typed branch state for numbers, variables, operators, relations, constraints, and intermediate results. Engrams supply memory; DeepLoop carries thought; skill LoRAs supply learned procedures; expert branches execute and verify operations. | **LOCKED** |
| D-054 | Cross-shape SVD labeling uses normalized singular spectra, principal angles after projection into the common 2,048D frame, and shared anchor-response fingerprints. Operator/projector fingerprints resolve sign ambiguity, and nearly degenerate singular directions are labeled as one subspace. | **LOCKED** |
| D-055 | Persistent parameters are the DeepLoop core, block-specific shared factors, per-skill \(q_s\), per-skill/per-block private factors, expert parameters, and router anchors. Live \(h_t\), router gates, \(Ah_t\) bottleneck activations, branch registers, and loop deltas are temporary activations. | **LOCKED** |
| D-056 | The coding agent reports 77 passing tests in the newer composed-LoRA worktree, including four added slot-population and private-rank-mask tests. The older snapshot currently available in this workspace independently discovers 72 tests: 58 pass and 14 skip under CPU-only unittest. Synchronizing the exact 77-test tree and adding formal five-condition provenance tests remain the acceptance gate. | **LOCKED** |
| D-057 | This master file is updated after every Dendritron project turn. Settled decisions enter normative sections, open hypotheses enter the open-decision ledger, implementation/test reports enter the evidence ledger, and superseded claims remain visible as REVOKED history. | **LOCKED** |
| D-058 | Stage 1 punctuation-aware v2 reconciliation and Stage 2 donor-row migration are complete. The audit retained 421,332 bigrams and 388,443 trigrams; the corrected donor bank contains 809,775 exact-identity copied rows plus 190,225 freshly extracted rows, totaling 500,000 bigrams and 500,000 trigrams with paired layers 8 and 24. | **LOCKED** |
| D-059 | HarMax, memory fusion, and LNGram are state-dependent nonlinear maps, so their evaluation order is an architectural choice. A final order requires an explicit decision and a controlled ordering ablation under O-015. | **LOCKED** |
| D-060 | Adapter and LNGram dependency tests separate direct dependency from mediated dependency. With \(h_t\) held fixed, perturbing raw donor payloads leaves adapter routing and \(W_q(h_t)\) unchanged. End to end, an active memory gate may influence both modules through the permitted residual change in \(h_t\). | **LOCKED** |
| D-061 | Residual-memory acceptance combines a zero-gate isolation test with an injection-boundary equation test. At zero gate, memory values and memory-projection parameters have zero influence on the residual update while the gate may retain a learning signal. At the injection boundary, the implementation must equal \(h+\gamma\odot\Delta m\) before the declared normalization. | **LOCKED** |
| D-062 | Frozen-donor acceptance requires three checks across forward, backward, optimizer step, and recurrent unroll: donor rows use frozen tensor/buffer storage, donor identities are absent from optimizer parameter groups, and exact tensor bytes or cryptographic hashes remain unchanged. Trainable JTD projection maps remain a separate parameter class. | **LOCKED** |
| D-063 | Procedural-SVD input consists of verified transition records with \(\Delta h_t=h_{t+1}-h_t\), skill/block/round identifiers, success evidence, and ordering metadata. Acceptance includes stalled-state zero deltas, translation invariance under \(h_t\mapsto h_t+c\), verified-trajectory filtering, and an entry-point contract that accepts transition records. | **LOCKED** |
| D-064 | This master is a self-contained coding-agent handoff. The coding agent reads Section 20 as the current implementation delta, preserves all LOCKED runtime equations unless a newer decision explicitly supersedes them, implements only the listed additions, and returns exact file diffs plus raw test evidence for reconciliation into the same master. | **LOCKED** |
| D-065 | Procedural-discovery records use immutable detached tensor snapshots plus explicit task and trajectory provenance. `task_id` and `trajectory_id` identify the episode; `skill_ids` remains optional observed-routing metadata because initial SVD may discover skill slots after capture. A structured `success_evidence` payload authorizes the derived `success` verdict, and ordering is scoped within each trajectory. Procedural SVD consumes verified `delta_h`; `h_matrix` remains a diagnostic view. | **LOCKED** |
| D-066 | A trajectory is identified by the ordered pair `(task_id, trajectory_id)`. Per-trajectory counters, filtering, listing, and uniqueness checks use that pair so different tasks may safely reuse local names such as `traj-001`. The canonical verified procedural-SVD path requires explicit task/trajectory identity and structured success evidence; compatibility defaults remain tagged legacy input until upgraded to explicit provenance. | **LOCKED** |
| D-067 | Every proposed project change receives one of three labels: **Architecture/Intent**, **Implementation/Verification**, or **Asset/Configuration**. Architecture/Intent changes require an explicit user decision before they become LOCKED or reach code. Each coding-agent handoff and review states **Changed**, **Preserved**, **Evidence status**, **Open risk**, and **Approval required**. Implementation corrections that enforce an existing LOCKED contract preserve the architecture and are identified as such. | **LOCKED** |
| D-068 | Geometric attention, HarMax/Harmonic Loss geometry, LNGram, and JTD are protected architecture invariants. JTD preserves the frozen layer-2 definition frame and aligns layer-8, layer-24, and live states into it. LNGram derives exact discrete latent n-gram addresses from the evolving live state. HarMax assigns scale-invariant inverse-Euclidean mass over a causal contraction pool; the harmonic residual and its negative derivative supply signed attraction and repulsion. Any substitution or removal is an Architecture/Intent decision under D-067. | **LOCKED** |
| D-069 | Sparse candidate retrieval and continuous geometric scoring are separate operations. Surface 3→2→1 lookup and LNGram exact definition addressing populate a bounded semantic-memory pool; Euclidean/HarMax geometry then scores and moves among those definition/phrase candidates. Independently, skill adjacency selects procedural experts and their branch pools. The operational form is exact address-and-score with local Euclidean definition-cone expansion. JTD uses source-specific learned alignment maps \(J_8,J_{24},J_h\); layer-2 definition rows remain the identity reference. Orthogonality, isometry, and metric preservation become claims only when imposed by the fitter and verified numerically. | **LOCKED** |
| D-070 | HarMax is exactly invariant to uniform scaling of all supplied distances, \(p(cd)=p(d)\). With \(d_{iq}^2=\lVert z_i-c_{iq}\rVert^2+\lambda_p\delta_{iq}^2+\epsilon^2\), coordinate-only scaling leaves \(\lambda_p\delta^2\) and \(\epsilon^2\) fixed, so the completed distances generally scale nonuniformly. Full coordinate-scale invariance therefore requires dimensionless normalized auxiliary terms or a common scaling rule; otherwise the invariant claim applies to the HarMax normalization over its completed distance inputs. | **LOCKED** |
| D-071 | Related-concept expansion uses bounded offline Euclidean adjacency in the fitted JTD concept-cone frame. Each concept anchor stores a configurable list of related anchor IDs plus construction distance and provenance. Runtime exact surface/JTD/LNGram addressing selects seed anchors; Gather-then-Compute unions their adjacency lists, applies causal/source validity filters, deduplicates candidates, and forms the contraction pool. The current live query recomputes its own Euclidean distance to every gathered candidate, HarMax assigns inverse-distance mass, and the Harmonic Loss derivative supplies attraction and repulsion. Offline anchor-to-anchor distances construct adjacency; live query-to-anchor distances control each recurrent visit. | **LOCKED** |
| D-072 | Runtime HarMax center semantics preserve the frozen-memory/live-compute split. A concept cone supplies a nonnegative family of semantic directions, while HarMax operates on finite anchor points in that aligned geometry. The live state, after its declared JTD alignment, is the center of the current distance measurement and the moving query. Candidate concept anchors are finite class-center landmarks. Evidence construction supplies target mass \(y_{iq}\): supported and related anchors receive positive mass, while opposing or unrelated candidates receive zero target mass and remain in the HarMax denominator. HarMax supplies current proximity mass \(p_{iq}\), and \(y_{iq}-p_{iq}\) determines signed force. The derivative with respect to the live state injects related anchor directions as attraction and competing anchor directions as repulsion while frozen anchor coordinates retain their stored values. Experience-tier centers enter only through verified offline creation or consolidation. | **LOCKED** |
| D-073 | A HarMax equilibrium satisfies zero net HarMax force, \(\sum_q(y_{iq}-p_{iq})(c_{iq}-u_i)/D_{iq}=0\). That balance may occur at a selected anchor or at an interior compromise among several anchors. A full DeepLoop fixed point additionally includes memory, LNGram, branch, skill/expert, residual-scaling, and RMSNorm contributions. Average exact hash/table address lookup is constant per fixed address; concept expansion and Euclidean/HarMax scoring cost \(O(Q_i d)\), with \(Q_i\) bounded by the configured local pool. The intended operator claim is zero pairwise query-key bilinear scoring, subject to the O-017 skill-router reconciliation. | **LOCKED** |
| D-074 | Surface phrase retrieval and constituent semantic seeding coexist. For every complete word observed in the input or decoded reasoning trace, the surface/JTD path emits its dictionary sense IDs as semantic seeds even when a longer exact bigram or trigram supplies the primary phrase-memory row. The 3→2→1 longest-match result remains the phrase-memory result; constituent sense seeds independently enter bounded concept-cone expansion, are deduplicated by stable sense ID, and retain source/span/order provenance. Raw word strings, tokenizer IDs, word IDs, and sense IDs remain lookup and trace metadata outside the latent coordinate tensor; definition and live-state vectors carry the numerical semantics. Latent reasoning steps with no decoded surface word obtain seeds through LNGram. | **LOCKED** |
| D-075 | LNGram converts evolving contextual live-state evidence into exact latent addresses whose resolved values are single-word dictionary sense anchors plus provenance. Surface-derived and LNGram-derived definition senses enter the same bounded existing definition-memory gather, causal/source filter, deduplication, and Euclidean/HarMax contraction path. Shared routing for synonyms, paraphrases, and related latent patterns is evaluated by route agreement, definition-sense recall, perturbation stability, address utilization/entropy, and collision statistics. The continuous live state remains the reasoning carrier while the discrete LNGram address remains a sparse definition-memory handle. | **LOCKED** |
| D-076 | Coordinate accounting distinguishes the canonical continuous frame from every auxiliary representation. Frozen layer-2 definition anchors, mapped layer-8/layer-24 memory, and the production live state use the 2,048D joint geometry. Hash addresses, LNGram integer addresses and bounded sense-handle lists, router gates, skill bottlenecks, branch registers, and metadata may use other shapes. `joint_to_live` is an independently learned movement map initialized as identity when square; inverse or cycle-consistent status requires an explicit fitter constraint and numerical evidence. SVD of verified \(\Delta h\in\mathbb R^{2048}\) produces activation-space procedural fingerprints, proposed slot labels, and router anchors; shared/private LoRA matrix factors arise through the subsequent coupled parameter/update fitter rather than by directly copying a right singular vector into a weight matrix. | **LOCKED** |
| D-077 | LNGram's complete architectural scope is latent lookup of word-definition memory. Its populated addresses resolve dictionary sense IDs, frozen layer-2 definition anchors, and bounded adjacency among those definition anchors. LNGram stores no skill IDs, expert IDs, skill-to-expert edges, branch specifications, or experience-record routes. Surface exact lookup continues to own layer-8/layer-24 phrase Engrams; Hash Engram continues to own canonical-ID local-pattern misses. | **LOCKED** |
| D-078 | An expert is a procedural task/skill/branch junction. Expert candidates originate from the selected skill's independent skill-to-expert adjacency and the requested task relation. Resolved definition evidence may condition or rank candidates after that bounded procedural adjacency is gathered, while the LNGram address itself carries no expert edge. | **LOCKED** |
| D-079 | Skill routing is independent of LNGram definition addressing. The live state is compared with populated procedural skill anchors under the final metric selected by O-017; each selected skill returns expert IDs from the skill-to-expert graph. LNGram contributes only definition-memory residual evidence to the live state, so any later effect on skills or experts is mediated through the permitted evolution of \(h_t\). | **LOCKED** |
| D-080 | Token and word IDs are symbolic lookup labels rather than coordinates in semantic geometry. Their integer values carry no distance meaning: IDs such as `bark=#123` and `tree=#932` may resolve sense anchors that lie close in the layer-2 definition frame. Polysemy remains sense-specific: the tree-covering sense of “bark” should occupy the tree/wood neighborhood, while the canine-vocalization sense should occupy the dog/sound neighborhood. Surface lookup may seed all recorded senses; contextual live-state LNGram addressing, bounded definition adjacency, and Euclidean/HarMax evidence determine their local influence. Raw IDs still serve input embedding, exact addressing, provenance, and output identity. | **LOCKED** |
| D-081 | **REVOKED:** A new version-coupled LNGram definition-address registry and separate registry builder are required. Replaced by D-082 after the user clarified that JTD and the completed definition assets already provide the shared semantic frame and source bank. | **REVOKED** |
| D-082 | The existing JTD, complete single-word layer-2 sense bank, and ordered definition-word graph are the semantic foundation for local generalization. Related-definition candidates come from those single-word sense anchors. Layer-8/layer-24 two-word and three-word Engrams retain a separate phrase-memory role because a phrase such as “tree bark” represents a narrower context than the broader “tree” neighborhood needed for a question about leaves. LNGram may use contextual live-state history according to its existing implementation while its retrieved semantic values remain single-word definition senses. The architecture reuses these existing assets as its sole semantic mapping and neighborhood source. | **LOCKED** |
| D-083 | JTD owns continuous mapping and semantic alignment; LNGram owns sparse pulling from that completed space. The bridge is handle-based: \(h_t\rightarrow W_q\rightarrow\) route symbols → exact latent address → one or more dictionary sense IDs → frozen layer-2 vectors. LNGram addresses, dictionary row IDs, and continuous vectors occupy separate namespaces. JTD begins with the fetched vectors and live query, while the integer address performs selection only. A populated address may pull a bounded candidate set so continuous Euclidean/HarMax evidence can resolve polysemy and routing collisions. The address-to-sense assignment objective remains O-019. | **LOCKED** |
| D-084 | LNGram sign thresholding provides deterministic exact lookup for a given learned bit sequence while semantic stability remains measured. Nearby projected states can cross zero hyperplanes and receive different codes; distinct states can share a code. Route offsets isolate route namespaces inside the physical table. Route agreement, projection margin, perturbation stability, candidate recall, and collision rate establish semantic routing quality. “Collision-free” applies specifically to the route-offset table layout. | **LOCKED** |
| D-085 | \(J_h\) denotes the live-to-joint linear map. Differentiating a scalar loss \(\rho(J_hh)\) with respect to \(h\) produces \(J_h^\top\) through the chain rule. Runtime joint-space movement uses the independently parameterized `joint_to_live` map \(J_{\mathrm{joint}\rightarrow h}\), initialized as identity when square. Transpose, inverse, and cycle-consistent status each require their corresponding explicit constraint and numerical evidence. | **LOCKED** |
| D-086 | Layer depth and runtime role remain separate distinctions. A frozen layer-2 definition row is a sense-specific lexical/compositional semantic landmark; the evolving DeepLoop state \(h_t\) carries contextual, procedural, and intermediate reasoning content. JTD aligns \(h_t\) with the landmark frame for distance evaluation. Claims such as tree-covering `bark` occupying the tree/wood neighborhood require measured sense-level distances and graph-neighborhood recall; the ordered definition-word graph supplies source-grounded candidate adjacency while Euclidean/HarMax geometry controls live influence. | **LOCKED** |
| D-087 | JTD maps before LNGram pulls. For the definition-memory path, the evolving state first becomes the joint-frame query \(u_t=J_hh_t\); LNGram then forms route logits and exact addresses from \(\operatorname{RMSNorm}(u_t)W_q\). This preserves live-state exclusivity because every address remains a function of \(h_t\), while keeping semantic alignment under JTD and selection under LNGram. | **LOCKED** |
| D-088 | A populated LNGram address resolves a bounded tensor record containing dictionary sense-row handles, validity/population masks, source provenance, and any calibrated evidence weights. The address integer and the sense-row integers retain separate namespaces. Frozen definition vectors remain the production semantic payload. Any compact continuous table retained to train the hard router serves only as a routing surrogate and requires exact agreement tests against the sense-handle path. | **LOCKED** |
| D-089 | Definition materialization respects Gather-then-Compute. SQLite metadata resolution, shard opening, and cache population occur outside recurrent block visits. Each live visit performs bounded tensor handle lookup, deduplication, and batched definition-row gather from a resident, memory-mapped, or explicitly managed hot-cache surface selected under O-016. | **LOCKED** |
| D-090 | The completed 1,532,746-row, 2,048D BF16 layer-2 definition bank is the frozen canonical landmark matrix. Its 31 safetensors shards and SQLite catalog are persistence and provenance assets. Runtime exposes the vectors once through a row-addressable resident, memory-mapped, or managed-cache tensor surface; the bank remains frozen conditional memory while \(h_t\) remains the evolving working state. | **LOCKED** |
| D-091 | JTD supplies semantic placement and LNGram supplies sparse access. The exact runtime bridge is latent address → bounded address record → sense-row handles → batched row gather from the already aligned definition bank. This bridge performs indexing rather than a second semantic map. Raw LNGram address arithmetic and dictionary-row arithmetic stay separate because route/order address spaces, polysemous senses, and bank rows have different cardinalities and meanings. | **LOCKED** |
| D-092 | In the signed definition field, \(y_{iq}\) and \(p_{iq}\) are scalar target-evidence and HarMax proximity masses over candidate anchors; \(c_{iq}\) and \(u_i\) are the anchor and live-query vectors. The force is \(n\sum_q(y_{iq}-p_{iq})(c_{iq}-u_i)/D_{iq}\). Repulsion occurs for candidates whose current mass exceeds target mass, especially nearby unsupported competitors; geometric distance alone does not assign the sign. Joint-space movement returns through the independently learned `joint_to_live` map under D-085. | **LOCKED** |
| D-093 | Model assembly attaches the complete frozen definition-bank tensor surface once. The recommended owner is an external frozen bank object or a nonpersistent CPU buffer with explicit device-aware batched gathers, so ordinary checkpoints do not duplicate approximately 5.8 GiB and generic model device moves do not silently migrate the entire bank. A raw LNGram address resolves a bounded sense-row list before `index_select`; direct equality between latent address and dictionary row remains excluded by D-091. | **LOCKED** |
| D-094 | Recovered 2026-08-12 execution records confirm that full JTD latent-asset export and real CPU fitting completed before recipient assembly. The phrase “memory bridge is built” in that checkpoint refers specifically to the fitted and frozen \(J_8\) and \(J_{24}\) donor-view projections. The same record explicitly names loading the fitted projections, surface index, token embeddings, and definition bank into Dendritron as the next assembly step. The absent complete bank tensor is therefore an unfinished recorded deliverable rather than a newly introduced architectural requirement. | **LOCKED** |
| D-095 | Production-width LNGram thresholding uses one learned \(W_q\in\mathbb R^{2048\times2048}\), reshaped into 512 four-bit routes. Each route emits one exact order-2 and one exact order-3 address. This supplies multi-route redundancy; the paper's optional multi-table construction is the distinct mechanism that would supply independently parameterized projections. Route offsets guarantee physical namespace separation only. Semantic aliasing (unrelated states sharing addresses) and fragmentation (related states crossing zero hyperplanes into different addresses) remain empirical properties of the learned projection and address population. Existing arithmetic and surrogate-gradient tests do not establish semantic stability. | **LOCKED** |
| D-096 | JTD and the definition field have complementary causal roles. `live_to_joint` aligns the evolving live state with frozen layer-2 definition coordinates, while `definitions_to_joint` is identity. The current `_definition_field` then forms inverse-distance weights, moves the joint query toward their weighted centroid, maps that movement through `joint_to_live`, gates it, and injects it as a residual. Thus the current landmarks actively attract the live state rather than merely score it. The locked complete operator replaces this attraction-only centroid with the signed HarMax \(y-p\) field under D-072/D-092, preserving positive-evidence attraction and unsupported-competitor repulsion. | **LOCKED** |
| D-097 | The rationale for combining JTD and LNGram is their complementary continuous/discrete boundary. The LNGram paper learns exact discrete routing keys from continuous hidden states, including hidden features originating in vision or multimodal backbones, so conditional lookup can operate without tokenizer-ID keys. The Locality Preserving Joint Transfer paper learns domain-specific continuous projections into a shared latent subspace. Dendritron adapts those roles as \(h_t\xrightarrow{J_h}u_t\xrightarrow{W_q,\operatorname{sign}}g_t\rightarrow\) bounded definition-sense handles \(\rightarrow c_q\xrightarrow{\mathrm{HarMax}}\Delta u_t\xrightarrow{J_{joint\rightarrow h}}\Delta h_t\). The live state and definition vectors remain continuous; only the sparse selection handle is discrete. Current fitted assets align layer-8, layer-24, and the recipient live view with the layer-2 definition frame. A future image or other modality receives its own fitted source-to-joint projection using paired, labeled, or distribution/geometry-preserving cross-domain evidence before sharing the definition-address system. | **LOCKED** |
| D-098 | Definition-bank attachment has one physical runtime owner and stays outside ordinary model checkpoint payloads and generic accelerator migration. Preferred ownership is an external frozen bank service/object; an in-module option uses a nonpersistent CPU-held tensor with explicit bounded device transfer and special handling that preserves CPU residence across `Module.to(...)`. `DendritronLM` may expose a loader/property while `SparseMemoryFusion` owns the tensor, or the inverse, but both modules do not register the full matrix independently. For 1,532,746 BF16 rows at width 2,048, the logical bank contains 6,278,127,616 bytes (approximately 5.85 GiB). Active sense rows are range-validated; masked sentinel rows gather safely. `sense_rows` → `index_select` → frozen definitions completes the materialization half of D-091. LNGram address → bounded address record → `sense_rows` remains the separate O-019 bridge. | **LOCKED** |
| D-099 | PyTorch buffer persistence and device management are separate controls. `register_buffer(..., persistent=True)` places the tensor in `state_dict()` and every ordinary checkpoint. `persistent=False` excludes it from `state_dict()` while retaining buffer participation in `_apply`, so `model.to("cuda")` still moves it unless ownership or transfer behavior is explicitly separated. The production definition bank therefore uses an external CPU owner or a tested custom CPU-resident attachment; only the bounded gathered rows cross to the live compute device. | **LOCKED** |

---

## 4. Current Project Status

| Stage | Artifact | Status | Evidence |
|---|---|---|---|
| 1 | Science-first 200M-word corpus; top 500k bigrams and top 500k trigrams | **IMPLEMENTED** | Punctuation-aware v2 recount and exact-row reconciliation are complete: 421,332 bigrams and 388,443 trigrams were retained by exact identity; 78,668 bigrams and 111,557 trigrams were replaced. |
| 2 | Frozen Qwen donor Engrams | **IMPLEMENTED** | Corrected migration is complete under the canonical punctuation-v2 root: 809,775 paired rows copied and 190,225 paired rows freshly extracted, totaling 1,000,000 rows with full 2,048D BF16 layer-8/layer-24 values. |
| 3S | Canonical definition-source acquisition and coverage | **CODED** | OEWN 2025+, Wiktionary 2026-07-06, and MeSH 2026 streaming parsers; source/license/hash contracts and synthetic parser tests pass. Real CPU source build and coverage report are next. |
| 3A | Dictionary inventory and definition-word graph code | **CODED** | `build_dictionary_inventory.py`; finalized source-manifest verification, synthetic polysemy, and definition-link tests pass. |
| 3B | Layer-2 dictionary donor bank | **IMPLEMENTED (USER-REPORTED)** | Recovered/current coding-agent reports identify 1,532,746 frozen 2,048D BF16 sense rows across 31 shards, approximately 5.8 GiB total, with SQLite metadata, donor revision `995ad96eacd98c81ed38be0c5b274b04031597b0`, and definition manifest SHA-256 `421564dfab6dffaa602fdc40675b7ae2d5864f51ab6beda21f5b1e445621fe33`. Definition transform is identity. |
| 3C | Optional definition-vector PCA diagnostic | **CODED** | `build_universal_subspace.py`; CPU-only diagnostic excluded from the required model path. |
| 3D | LNGram latent-address module | **CPU VERIFIED / DEFINITION-BANK INDEX BRIDGE AND ROUTE-STABILITY EVIDENCE REQUIRED** | Exact 2/3-gram route addresses and one-bit counterfactual routing gradients pass CPU tensor tests. At width 2,048, the single learned projection yields 512 four-bit routes and up to 1,024 route-local addresses per position across orders 2 and 3. The current paper-native learned width-4 value tables require realignment under D-077/D-091: production addresses select bounded sense-row records, and batched gathers return the existing frozen layer-2 rows. O-019 governs the replacement or adaptation of the current value-difference routing gradient; O-022 governs threshold stability and bounded aggregation across the 512 routes. |
| 3E | Longest 3→2→1 runtime resolver | **CODED / PAYLOAD MATERIALIZATION VERIFICATION REQUIRED** | `dendritron/retrieval.py` exposes variable recipient-token spans, trigram/bigram priority, one-word polysemy preservation, and lower-order decomposition candidates. Acceptance now traces those candidate sense IDs through payload construction into the actual definition tensor consumed during recurrent execution. |
| 3F | Expert junction graph | **CONTRACT CORRECTION REPORTED / LIVE ROUTER SEPARATE** | Coding-agent report: `ExpertGraph.route()` now gathers candidates only from `skill_to_experts`; concept IDs remain ranking metadata, and two regression tests bring the focused file to 4/4 passes. Call-site inspection found this class used only by tests; live execution routes through `SkillExpertSystem` and `ComposedWorkingLoRA`, so the edit enforces the stored graph contract while leaving runtime behavior unchanged. |
| 4A | Locked Qwen tokenizer and canonical projection contract | **CODED / LOCAL TOKENIZER ASSET REPORTED** | `dendritron/tokenizer.py` resolves the exact Stage-2 revision, implements the official Engram normalizer, fingerprints projection P, measures Qwen's effective vocabulary, and verifies case/space/accent/whitespace equivalence while retaining distinct punctuation classes. Coding-agent report: a `qwen-tokenizer/` asset with `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and 248,077 vocabulary entries is saved at the competition-project root. Exact revision, complete file inventory, hashes, and projection-fingerprint agreement enter the local asset verification gate. |
| 4B | Surface-memory compiler and runtime index | **CODED** | `stage3_jtd/build_joint_transfer_domain.py` and `dendritron/jtd.py`; canonical token hashes, exact tuple verification, complete-word offset fallback, tokenizer/projection fingerprints, punctuation barriers, and 3→2→1 runtime lookup pass synthetic integration tests. The historical filename remains for compatibility. |
| 4C | Frozen Engram payload loader | **CODED** | `dendritron/engram_store.py`; maps the selected row back to the immutable layer-8/layer-24 shard with hash validation and lazy caching. |
| 4C-D | Frozen definition payload loader | **ATTACHMENT AND SENSE-ROW GATHER REPORTED / STORAGE CONTRACT CORRECTION REQUIRED** | Coding-agent report: `DendritronLM.load_definition_bank`, `SparseMemoryFusion.attach_definition_bank`, and `_definition_field` now attach a complete tensor and gather frozen rows from supplied `definition_sense_rows`; 11 new tests and 47 total tests pass under an unspecified command. This completes supplied-sense-row materialization while LNGram address population remains pending. The report describes a persistent buffer registered through both model and fusion surfaces, conflicting with D-093/D-098 because it enters checkpoint/device traversal and creates dual ownership of an approximately 5.85 GiB asset. Acceptance requires one owner, checkpoint exclusion, controlled CPU residence/bounded device transfer, active-row upper-bound validation, the exact test command, and reconciliation with the previously reported 136-test suite. |
| 4D | Trainable Hash-Engram miss addressor | **CODED** | `dendritron/hash_engram.py` and `dendritron/memory_pipeline.py`; deterministic multi-head canonical-Qwen-ID hashes retain punctuation and activate on frozen donor misses while dictionary candidates survive. |
| 4E | Layer-2 joint concept frame | **CPU VERIFIED / SIGNED-FIELD INTEGRATION REQUIRED** | `dendritron/joint_transfer.py` correctly keeps definitions as identity anchors and supplies layer-8/layer-24/live maps plus an independent joint-to-live movement map. Current `memory_fusion._definition_field` performs inverse-distance centroid attraction; D-032/D-072 require target mass (y), HarMax proximity mass (p), and the signed (y-p) field for attraction and repulsion. |
| 4F | Qwen symbol seed and real JTD anchors | **IMPLEMENTED (USER-REPORTED)** | Recovered completion record: full export at `/data/dendritron-stage4-jtd-punctuation-v2/latent-assets`; 248,077×2,048 BF16 token embeddings; 32,768 bigram and 32,768 trigram anchor triples; layer-2 reference, layer-8/layer-24 sources; source-contract fingerprint `a1a284b0d053e6be767dd14fb3a7f4cd618e8796554ddcda0fb373c0213f2d98`. Token embedding SHA-256 is `0cd940a9a4fcfab9d7486f7ada6ad13b69d821bdef38789c2788d12ad6583b7e`; bigram/trigram anchor SHA-256 values are `f3627d57f2c83b72a62967fb052d575dd5efc7b4f40f03481c9da83c1997ae04` and `91e4815eae0ec58bafa7b6ffc74e70d4dd93dae1a615321e9bb5c8d0fbcc6ecc`. |
| 5 | JTD and sparse memory-fusion maps | **REAL CPU FIT COMPLETE (USER-REPORTED)** | Recovered schema-v2 completion report: 65,536 paired anchors per source; \(J_8\) loss 21.849→0.372 and \(J_{24}\) loss 37.923→0.844; checkpoint `/data/dendritron-stage4-jtd-punctuation-v2/jtd-projections.pt`; joint/live width 2,048; definitions identity; surface metadata excluded from vectors; live map identity-initialized for recipient training. |
| 7 | Two-block HarMax recurrent core | **CPU VERIFIED** | Both stored blocks execute two sequential Post-RMSNorm sublayers and backpropagate through HarMax target/distance mass, attraction/repulsion, and harmonic residuals. |
| 8 | Skill anchors, composed working LoRA, expert graph, and expert-owned branches | **CODING-AGENT CURRENT TREE / 136 TESTS REPORTED** | Earlier breakdown: 77 baseline runtime tests, 12 D-060/D-061/D-062 provenance tests, 12 D-063 transition tests, and 19 D-065 tests produced 120 passes. After later transition/provenance and expert-graph additions, the coding agent reports 136 passed and zero failed. The current tree remains the implementation authority for exact diffs and raw output. |
| 8P | Live-state provenance and procedural-transition capture | **D-065 CODED / D-066 PAIR-KEY CORRECTION NEXT** | Coding-agent report: `SuccessEvidence`, task/trajectory fields, evidence-derived success, optional empty skill IDs, detached clones, filtered matrix views, listing methods, and 19 focused tests are committed. Per D-066, counters and exact trajectory operations now need `(task_id, trajectory_id)` keys plus an explicit-provenance canonical SVD gate. |
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
| LNGram addressing projection \(W_q\) | Trainable in base/offline learning windows; frozen at runtime |
| Block-specific shared LoRA factors | Frozen at runtime; revised only by controlled consolidation |
| Per-skill shared coefficients \(q_s\) | Persistent and frozen at runtime |
| Per-skill/per-block private LoRA factors | Persistent; selected skill updates only in a validated offline buffer |
| Committed expert knowledge/task/skill junctions and branch specifications | Frozen after commit |
| Skill-to-expert adjacency | Appendable after successful episodes |
| Routing/skill priors and experience Engram tier | Appendable after verification |

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
| LNGram definition lookup | Discrete symbols learned from the current hidden state | Retrieves dictionary sense records and bounded definition adjacency | After an initial hidden state exists |
| Skill graph | Selected skill IDs | Returns the expert IDs already adjacent to each skill | After skill routing |
| Experience Engram lookup | Task, trajectory, skill, expert, and outcome IDs | Retrieves successful trajectory traces, outcome records, and optional routing/skill priors | After skill and expert resolution |

Qwen token hashing and LNGram share the general idea of constant-time discrete
addressing. Their inputs and tables stay separate:

```text
raw Qwen IDs -> frozen P -> surface exact buckets / Hash Engram rows
live hidden states -> LNGram symbols -> dictionary sense rows
```

### 7.1 LNGram address construction

Here \(h_t\) is the current 2,048D recipient DeepLoop state produced from the
initial input state and every completed recurrent visit. Frozen donor
layer-2/layer-8/layer-24 rows remain separate memory objects. JTD first places
the current live state in the canonical layer-2 definition frame:

\[
u_t=J_hh_t,\qquad
r_t=\operatorname{RMSNorm}(u_t), \qquad
\ell_t=r_tW_q.
\]

Channels are divided into routes. Each route is hard-discretized into a symbol:

\[
a_{t,r}
=
\sum_{j=0}^{M-1}
\mathbf{1}[\ell_{t,(r,j)}>0]2^j
\]

An n-gram ending at \(t\) becomes an exact latent address:

\[
g_{t,r}^{(n)}
=
rK^n+
\sum_{i=0}^{n-1}
a_{t-n+1+i,r}K^i
\]

This is the LNGram mechanism retained from the attached paper: hidden state to
discrete symbols to exact latent n-gram lookup. Exactness means that a given
bit sequence selects one deterministic table address. Semantic neighborhood
quality remains an empirical property of the learned \(W_q\): nearby projected
states can cross a zero threshold, and distinct projected states can share a
code. Route offsets isolate the physical route namespaces. Margin, route
agreement, perturbation stability, recall, utilization, and collision
statistics are required evidence.

This ordering is the reason JTD and LNGram coexist. JTD aligns continuous
views; LNGram derives sparse discrete handles from the aligned continuous live
view; the handles gather continuous definition anchors; and HarMax evaluates
and moves within the shared continuous geometry. The production text assets
fit \(J_8\), \(J_{24}\), and initialize/train \(J_h\). A future vision path
would add a separately fitted \(J_{vision}\) before it participates in the
same address and definition-memory system.

### 7.2 Dendritron definition-address and adjacency record

JTD has already mapped the continuous definition geometry. LNGram supplies a
latent address that pulls handles from that completed space. Dendritron assigns
each populated address the following definition-memory schema:

```text
definition_address_record
  lngram_address
  dictionary_sense_ids[]
  frozen_layer2_row_ids[]
  adjacent_definition_sense_ids[]
  source_provenance[]
  route_statistics
```

The address value and dictionary row number belong to separate namespaces. The
exact address selects this record; its sense IDs fetch the frozen layer-2 rows;
the fetched rows enter the existing definition gather and HarMax pool. This
handle-based bridge keeps the continuous representation entirely in the frozen
definition vectors and leaves semantic-mapping ownership with JTD. The current
hidden state supplies the context needed to select among polysemous definition
neighborhoods. Integer token and word IDs serve lookup and provenance;
Euclidean distance is computed between the frozen sense anchors and the
JTD-aligned live state.

Synonym and paraphrase sharing is learned and measured from the binary address
construction. Required diagnostics include route agreement, definition-seed
recall, bit/address stability under meaning-preserving perturbations, route
utilization and entropy, and collision rates.

Surface retrieval supplies a second seed source. A longest 3→2→1 match selects
the primary phrase-memory row, while every complete constituent word also
contributes its dictionary sense IDs to the semantic seed set. Stable surface,
token, word, and sense identifiers remain metadata and provenance; the JTD
definition anchors and live state remain the numerical latent coordinates.

The geometric router evaluates the small bank of principal skill anchors. Once skill \(s\) is active, its adjacency record supplies:

\[
\mathcal E_s
=
\{e:\,(s,e)\in G_{\text{skill-expert}}\}
\]

An expert can have edges to several skills. Hundreds of expert nodes can therefore occupy one skill neighborhood while a composite expert can sit near several skills.

### 7.3 Definition pull and HarMax readout

Surface or LNGram handles pull dictionary sense IDs. Those IDs fetch frozen
layer-2 anchors \(c_q\) from the already constructed JTD definition space. The
live query in that same space is

\[
u_t=J_hh_t.
\]

For a bounded, causally valid definition pool:

\[
D_{tq}
=
\frac{1}{d}\left\|u_t-c_q\right\|_2^2
+\lambda_p\delta_{tq}^2+\epsilon^2,
\qquad
p_{tq}
=
\frac{D_{tq}^{-n/2}}
{\sum_jD_{tj}^{-n/2}}.
\]

Verified source, definition-graph, and task evidence constructs normalized
target mass \(y_{tq}\). The signed joint-space pull is

\[
\Delta u_t
=
n\sum_q
(y_{tq}-p_{tq})
\frac{c_q-u_t}{D_{tq}}.
\]

The runtime movement map and residual injection are

\[
m_t=J_{\mathrm{joint}\rightarrow h}(\Delta u_t),
\qquad
h_t^{\mathrm{mem}}
=
h_t+\gamma_t\odot m_t.
\]

Here \(J_h\) performs live-to-joint alignment and
\(J_{\mathrm{joint}\rightarrow h}\) performs runtime movement back to the
recipient. Differentiating a training loss \(\rho(J_hh_t)\) with respect to
\(h_t\) produces \(J_h^\top\) through the chain rule; that transpose is distinct
from the independently parameterized runtime movement map.

where \(\gamma_t\) is a bounded learned or calibrated gate. Every retrieved
dictionary sense remains a distinct frozen point with continuous mass, while
\(h_t\) remains the live recurrent state. Section 9 supplies the same
target-minus-distance derivative for general contraction pools.

### 7.4 Search contract

Runtime routing follows this order:

1. Exact 3→2→1 surface resolution selects the primary phrase-memory row and emits every complete constituent word's dictionary sense IDs as semantic seeds.
2. LNGram resolves one or more latent addresses into dictionary-sense seed and definition-adjacency records.
3. Surface and LNGram definition seeds gather bounded definition adjacency, apply causal/source filters, and deduplicate by stable sense ID.
4. Live Euclidean/HarMax scoring forms the active definition contraction pool.
5. Geometric routing scores the populated skill registry.
6. Each selected skill returns its stored expert adjacency.
7. The requested task relation and resolved definition evidence condition or rank the already bounded skill-adjacent expert set.
8. Each active expert instantiates its own branch activations in the current physical block.

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

> **Coordinate-binding gate:** D-069 governs pool construction, and O-017
> governs the final coordinate notation and skill-router equation. Every live
> query and candidate point must enter the same JTD frame before Euclidean
> distance is evaluated. The current \(h_i\)/\(z_{iq}\) display below remains
> the executable historical form until O-017 is explicitly resolved; new
> router work uses the guarded formulation recorded in O-017.

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
  sense retrieved by surface or LNGram address, branch anchor, or
  branch-generated candidate;
- \(\omega_{iq}\geq 0\) is its evidence strength;
- \(\sigma_{iq}\in\{+1,-1\}\) marks supported versus opposing evidence;
- \(\pi_{iq}\) retains the exact source span, memory row, concept, expert, and
  branch pointer.

The pool is causally bounded. Hidden-state evidence comes from positions
\(j\leq i\), and a memory or definition anchor can enter only through a source
span ending at or before \(i\). Sparse surface/JTD/LNGram definition retrieval,
selected skill adjacency, and active expert branches bound their respective
pool contributions to \(Q_i\); a global state-to-state comparison is
unnecessary.

Related-definition expansion follows D-071's **seed-and-expand** contract.
Exact surface/JTD/LNGram addresses answer **where and when** by selecting
definition-sense seed anchors. Each seed contributes a bounded offline-built
Euclidean adjacency list from the aligned definition cone. Gather-then-Compute unions the lists,
applies causal and source-validity filters, and deduplicates the candidates.
The evolving live query then recomputes Euclidean distance and HarMax mass over
that local pool to determine **which related definitions and how strongly** they
participate. The inverse-distance normalization and Harmonic Loss derivative
make this second stage nonlinear in the live state while preserving a sparse
runtime working set. O-018 governs adjacency fanout and construction policy.

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

The live-point and class-center roles follow D-072. The aligned live state is
the center from which current distances are measured and the variable updated
by the derivative. The concept cone supplies nonnegative combinations of
semantic directions; concept anchors supply the finite class-center landmarks
used by HarMax.
HarMax supplies the current proximity distribution \(p_i\), while causal
evidence, exact addresses, branch bindings, and verification supply the target
distribution \(y_i\). With one-hot \(y_i\), one anchor is the target center.
With distributed \(y_i\), several related anchors define a local target region.
Its evidence centroid

\[
\mu_i=\sum_{q=1}^{Q_i}y_{iq}z_{iq}
\]

summarizes that region, while the executable force remains the exact
multi-anchor inverse-distance derivative below. Runtime attraction means that
related anchor directions enter the residual update of the live state; frozen
memory coordinates retain their stored values. D-073 governs the distinction
between HarMax force balance and full recurrent convergence.

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

### 10.3 Separate SVDs for semantic memory and procedural skill

SVD is both a capacity estimator and a label proposer, with two distinct input
families.

Semantic analysis operates on centered definition or Engram rows:

\[
M_{\text{semantic}}
=
\operatorname{Center}
\left(
\{m_i^{(2)},m_i^{(8)},m_i^{(24)}\}
\right)
\]

Its singular modes describe organization in stored knowledge. They support
diagnostics, indexing, compression studies, and semantic labels.

Procedural analysis operates on successful computation evidence:

\[
D_{\text{proc}}
=
\operatorname{Stack}
\left(
\{\Delta h_t,\Delta o_t,\operatorname{vec}(\Delta W_t),J_t\}_{\text{verified}}
\right)
\]

\[
D_{\text{proc}}=U\Sigma V^\top
\]

where \(\Delta h_t=h_{t+1}-h_t\), \(\Delta o_t\) is a verified operator
movement, \(\Delta W_t\) is a successful adapter update, and \(J_t\) is an
optional Jacobian sketch. Retained procedural modes propose skill slots and
router anchors. The retained count follows measured cumulative energy,
reconstruction error, task success, and replay preservation.

A principal direction is the dominant fingerprint used to discover, label,
route, and compare a skill. Every routed skill receives the complete block
state and all variance present in that state.

Across matrices of different sizes, compare:

- normalized singular spectra;
- principal angles after projection into the common 2,048D frame;
- responses to the same anchor suite;
- operator or projector fingerprints, which are stable to singular-vector
  sign flips;
- nearly degenerate directions as a grouped subspace.

This keeps semantic-memory labels and procedural-skill labels separate while
allowing both analyses to use SVD.

### 10.4 Composed shared/private skill LoRA

Let \(K\) be the shared-basis capacity and \(r_s\) the private rank for skill
\(s\). For physical block \(b\), store:

\[
A_b^{\mathrm{sh}}\in\mathbb{R}^{K\times d_{\mathrm{in}}},
\qquad
B_b^{\mathrm{sh}}\in\mathbb{R}^{d_{\mathrm{out}}\times K}
\]

For each populated skill \(s\), store:

\[
q_s\in\mathbb{R}^{K},
\qquad
A_{s,b}^{\mathrm{priv}}\in\mathbb{R}^{r_s\times d_{\mathrm{in}}},
\qquad
B_{s,b}^{\mathrm{priv}}\in\mathbb{R}^{d_{\mathrm{out}}\times r_s}
\]

The complete skill adapter is:

\[
\boxed{
\Delta W_{s,b}
=
B_b^{\mathrm{sh}}
\operatorname{diag}(q_s)
A_b^{\mathrm{sh}}
+
B_{s,b}^{\mathrm{priv}}A_{s,b}^{\mathrm{priv}}
}
\]

Equivalently, its shared term is
\(\sum_{j=1}^{K}q_{s,j}b_{b,j}a_{b,j}^{\top}\). The coefficient vector \(q_s\)
is shared across blocks to preserve skill identity. The shared factors are
block-specific so Block 1 expansion and Block 2 contraction can perform
different transformations for the same skill.

For router activations \(g_{s,t}\), the block applies:

\[
\Delta h_{b,t}^{\mathrm{skill}}
=
\sum_{s\in\mathcal S_t}
g_{s,t}\Delta W_{s,b}h_{b,t}
\]

The contract is:

- A task may activate several populated skill slots at once.
- Every activated skill consumes the complete \(h_{b,t}\) as full-state input.
- \(q_s\) may be dense across the complete frozen shared basis.
- Every skill's private \(A\) has all \(d_{\mathrm{in}}\) columns and every
  private \(B\) has all \(d_{\mathrm{out}}\) rows.
- Rank counts bottleneck channels, equivalently the maximum number of
  independent outer-product terms represented by that factor pair. Words,
  steps, skill count, and visible hidden dimensions remain separate quantities.
- One rank-32 adapter can algebraically span as many as 32 matrix directions.
  Separate per-skill adapters add routing, reuse, expert attachment, isolated
  offline updates, and individual validation.

The current 21 skill slots are implementation capacity awaiting measured
population. The initial capacity target is \(S_{\max}=32\) with a populated
mask and active mask. If procedural SVD retains 32 meaningful modes, up to 32
skill slots may receive labels and complete adapters. A 16-component shared
basis is an initial hypothesis when measurements show that roughly half of
those modes recur across tasks; it remains subject to principal-angle,
explained-energy, and replay tests. Shared components count reusable
procedural structure, rather than a separate class of “shared skills.”

Private rank begins as a padded/masked implementation setting and is selected
per skill from residual SVD energy. The current value \(r_s=4\) is provisional.

### 10.5 Experts are knowledge/task/skill junctions

An expert records a reusable interaction between active knowledge, a requested
relation, useful principal skill directions, and branch wiring:

\[
e_j
\equiv
\left(
\mathcal K_j,
\mathcal T_j,
\mathcal S_j,
\mathcal B_j,
p_{g,j},
\text{success statistics}
\right)
\]

where:

- \(\mathcal K_j\) contains linked dictionary-sense and Engram IDs.
- \(\mathcal T_j\) identifies the task relation or requested outcome.
- \(\mathcal S_j\) is the sparse set of adjacent populated skills.
- \(\mathcal B_j\) is the persistent set of branch specifications.
- \(p_{g,j}\) is an optional reference to a successful routing/skill prior in
  the experience tier.

The expert's identity is the cross-interaction junction. The prior can
seed router gates and skill selection while remaining a separate record. One
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
- \(m_{e,j}\) is the sparse mask over populated skills;
- \(\boldsymbol\sigma_{e,j}\) assigns supported/opposing signs to the bound
  roles;
- \(\mathcal A_{e,j}\) stores exact dictionary, Engram, task, and relation
  anchors. LNGram may retrieve a dictionary anchor before branch binding while
  its address remains outside the expert record.

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
M_{\text{sense}}^{\text{surface}\cup\text{LNGram}},
g_{\text{skill}}^{(r,k)},
p_{g,e}
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
skill adjacency, requested task relation, and stored success evidence.
Surface/LNGram-resolved definition evidence may condition or rank that already
bounded expert set. Branch specifications remain attached to their expert junctions; contraction pools,
movements, residuals, evidence, and traces are recreated on every visit.

### 10.7 Skill-first routing

For current state \(z\), geometric routing scores the populated skill anchors:

\[
g_s
=
\widehat z^\top M_s\widehat S_s
\]

Select the active skill set by \(|g_s|\), subject to the populated mask. Then
retrieve the associated experts:

\[
\mathcal E_{\text{candidate}}
=
\bigcup_{s\in\mathcal S_{\text{active}}}
\mathcal E_s
\]

The requested task relation, stored success metadata, and resolved definition
evidence reduce this local set to the active experts. LNGram contributes the
definition evidence while its address carries no expert edge. This is
adjacency resolution around selected skills rather than a global geometric
comparison against every expert node.

### 10.8 Runtime freeze, offline learning, and expert formation

Runtime executes with persistent parameters frozen. It records live states,
selected skills, router gates, branch traces, typed registers, outputs, and
verification evidence. These are trajectory data rather than parameter
updates.

After successful verification:

1. Retain the successful trajectory and outcome evidence in the experience
   tier.
2. Resolve the active memory anchors, concepts, task relation, skills, and
   experts.
3. Copy the selected skill's block-specific private factors into an offline
   working buffer.
4. Train that buffer against the successful trajectory plus replay examples.
5. Run skill-task, cross-skill, memory, regression, and stability validation.
6. Commit and version the candidate private factors only after every required
   gate passes.
7. Reuse an existing expert junction when the cross-interaction is already
   represented, or commit a verified new junction and its graph edges.

The shared factors \(A_b^{\mathrm{sh}},B_b^{\mathrm{sh}}\), the DeepLoop core,
other skills, memory rows, and expert library remain frozen during an ordinary
private-skill update. Repeated validated residuals from several skills may
enter the controlled consolidation path in Section 10.9.

### 10.9 The x-part

For a verified procedural update signal \(\delta\) and current orthonormal
shared basis \(U\):

\[
c=U^\top\delta
\]

\[
x=(I-UU^\top)\delta
\]

- \(Uc\) uses existing shared procedural structure.
- \(x\) records private residual update energy and remains attached to the
  selected skill and its successful experience record.

Repeated, coherent x-parts indicate pressure for a new shared skill direction. Consolidation occurs when their energy and reuse exceed locked thresholds:

1. Collect validated x-directions that recur across several skills or tasks.
2. Merge them with the current basis and recompute a compact basis by
   SVD/HOSVD.
3. Select rank from explained energy, reconstruction error, task success,
   latency, and replay preservation within the configured capacity.
4. Reproject every \(q_s\) and affected private residual into the revised
   shared/private decomposition.
5. Validate all committed skills and freeze the revised basis for the next
   inference interval.

This separates three outcomes:

- Existing junction and useful prior: reuse the expert and its prior.
- New knowledge/task/skill cross-interaction: add an expert junction.
- Skill-specific residual: retain it in that skill's private factors.
- Repeated cross-skill residual: evaluate it for shared-basis consolidation.

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
\mathcal G^{(r,0)}
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
\mathcal G^{(r,1)}
\right)
\]

where \(H^{(0,0)}=H^{(0)}\). Thus:

\[
\text{stored depth}=2,
\qquad
\text{effective thought depth}=2R
\]

Each physical-block visit performs:

1. LNGram definition-address update
2. Surface/JTD/LNGram definition-memory resolution
3. Geometric skill selection
4. Skill-to-expert adjacency resolution
5. Expert-owned branch-pool construction
6. HarMax geometric/memory/branch contraction
7. First DeepLoop residual update and RMSNorm
8. Composed shared/private skill LoRA and expert-soma computation
9. Second DeepLoop residual update and RMSNorm
10. Successful-trajectory trace capture
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

The second sublayer applies the selected composed shared/private skill LoRAs
and the expert somas produced from those branch activations:

\[
E_k^{(r)}
=
\Delta_{\text{skill-LoRA}}^{(r,k)}
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
4. **Phrase and sense resolution:** at every complete-word endpoint select the
   primary exact trigram, otherwise bigram, otherwise one-word memory result;
   independently emit every complete constituent word's dictionary senses as
   semantic seeds.
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
10. **LNGram discretize:** live current state \(\rightarrow\) latent addresses,
    then resolve those addresses into frozen dictionary-sense memory and
    bounded definition-adjacency records.
11. **Skill routing:** compare the current state against the populated skill
    registry and produce sparse runtime gates.
12. **Expert adjacency:** selected skills \(\rightarrow\) adjacent expert IDs;
    the task relation and resolved definition evidence rank the bounded set.
13. **Branch binding:** instantiate each selected
    \(\mathcal B_{e,j}\), bind its semantic roles, preserve its exact pointers,
    and construct its causal contraction pool.
14. **First residual sublayer:** compute HarMax distance mass, signed
    attraction/repulsion, harmonic residuals, memory movement, and raw branch
    contraction; apply \(\operatorname{RMSNorm}(\alpha H+C)\).
15. **Second residual sublayer:** apply the composed shared/private skill LoRAs,
    combine branch movements through expert and block somas, and apply
    \(\operatorname{RMSNorm}(\alpha U+E)\).
16. **Block 1 expansion:** retain several relation and explanation proposals
    with their branch residuals and traces.
17. **Block 2 contraction:** test the proposals with opposing evidence,
    deduction, causal consistency, contrast, and counterfactual intervention;
    integrate the supported result.
18. **Deep loop:** return Block 2's hidden state to Block 1 and repeat for the
    round budget or convergence rule.
19. **Trace:** record live transitions, routed skills, branch registers,
    verifier evidence, and outcome while persistent parameters remain frozen.
20. **Offline learn and commit:** after verified success, append the trajectory
    experience; train the selected private skill factors in a replay buffer;
    validate and version the candidate; then reuse or append the corresponding
    expert junction and graph edges.
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
build path. Procedural explained-energy tests belong to verified transition and
adapter-update data in Stage 8.

### Stage 3D - Implement LNGram

Implemented module:

- `dendritron/lngram.py`
- 4-bit routes
- 512 routes at production width 2,048, generated by one learned projection
- exact 2/3-gram route-partitioned tables
- one-bit counterfactual routing gradient
- signed geometric readout

The route-partitioned layout is exact and collision-free at the physical table
namespace level. Semantic aliasing and threshold fragmentation remain measured
router properties. The definition-bank payload conversion also requires one
bounded aggregation rule across the 512 route outputs so route redundancy does
not expand the candidate pool without a global cap.

### Stage 3E - Implement longest memory routing and expert topology

Implemented modules:

- `dendritron/retrieval.py`
- exact word-order 3→2→1 resolution over variable recipient-token spans
- one-word sense preservation and lower-order decomposition access
- `dendritron/expert_graph.py`
- many-to-many knowledge/task/skill junctions
- optional routing/skill-prior references separate from expert identity

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
- train route-partitioned exact lookup values that resolve dictionary senses
- measure the implemented counterfactual surrogate gradient
- compare signed geometric readout with the paper's sigmoid readout

Acceptance test: latent-key consistency, table utilization, dictionary-sense
recall, contextual polysemy resolution, and performance on surface/JTD misses.

### Stage 7 - Add skills, experts, and branches

Include:

- procedural SVD/HOSVD slot proposal and cross-shape labels
- capacity bank, populated mask, and active mask
- block-specific shared LoRA factors
- per-skill shared coefficient vectors \(q_s\)
- per-skill/per-block private LoRA factors
- skill-first geometric router
- independent skill-to-expert adjacency
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

### Stage 9 - Activate offline private-skill learning, expert formation, and shared consolidation

Include:

- successful-trajectory collection and verifier contract
- selected-private-factor working buffer
- replay, regression, stability, and cross-skill validation gates
- versioned private-skill commit
- knowledge/task/skill junction novelty threshold
- optional routing/skill-prior attachment
- experience Engram append
- skill-to-expert edge updates driven by verified procedural evidence
- shared-span projection and private residual measurement
- temporary cross-skill residual directions
- consolidation thresholds
- shared-coordinate and private-residual reprojection tests
- forgetting and backward-transfer tests

---

## 15. Required Ablations

| ID | Configuration | Question |
|---|---|---|
| A0 | Geometric recurrent core | Can the core learn token transitions? |
| A1 | A0 + JTD frozen memory | Does exact grafted knowledge improve rank and reasoning? |
| A2 | A1 + LNGram | Does latent lookup improve misses and emerging patterns? |
| A3 | A2 + composed shared/private skill registry | Do reusable shared components plus isolated private residuals improve task transfer? |
| A4 | A3 + skill adjacency + fixed experts + expert branches | Does skill-local expert execution improve efficiency or quality? |
| A5 | A4 + two physical blocks at \(R=2,4,8\) | Does recurrent depth improve compositional reasoning? |
| A6 | A5 + verified trajectory experience, offline private update, and expert-junction commit | Do successful traces improve a selected skill while expert identity remains stable across related tasks? |
| A7 | A6 + cross-skill residual consolidation | Does continual growth add shared procedural structure while preserving committed skills? |

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
- Expert skill-edge sparsity and post-adjacency definition-conditioning rate
- Router-gate trace per block visit
- Branch contribution norms
- New expert-junction commit, reuse, and merge counts
- Routing/skill-prior reuse counts
- Experience Engram append count
- Skill-to-expert edge growth
- Loop delta \(\rho_r\)
- Visit alignment \(\kappa_R\)
- Skill-basis explained update energy
- private residual and cross-skill residual energy
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
| O-013 | Shared/private fitter identifiability | Preferred baseline: fit and gauge-fix the coupled shared basis across verified skills, project each target onto that span, then fit private factors only to the orthogonal residual. Compare against constrained joint fitting with private-product regularization and shared/private overlap penalties. Record reconstruction, shared energy, private energy, decomposition stability across seeds, task success, and replay preservation. |
| O-014 | Current-tree evidence boundary | The coding agent's current worktree supplies implementation paths, diffs, configuration, tensor shapes, tests, and raw output. This master supplies architecture and intent. The older workspace snapshot remains historical evidence and supplies zero path prescriptions for current changes. Each coding-agent response records its current HEAD/branch, exact changed paths, and raw tests so the master can reconcile behavior without transferring another project copy. |
| O-015 | Contraction-sublayer evaluation order | Compare parallel evaluation from one block input against HarMax→memory→LNGram and LNGram→memory→HarMax. Measure route stability, memory hit quality, recurrent convergence, task accuracy, and sensitivity across rounds, then lock the order explicitly. |
| O-016 | CPU Gather-then-Compute kernel | Verify the exact newer implementation, then benchmark gather-then-compute against grouped per-skill execution at matched outputs, sparsity, thread count, memory traffic, latency, and batch/sequence shapes before locking the runtime kernel. For definition memory, compare full resident BF16 storage, memory-mapped batched gather, and bounded hot-cache policies; keep SQLite metadata queries and shard-opening work outside recurrent visits. |
| O-017 | Meaning of \(z\) and the skill-routing metric | Exact addresses determine where and when, and Euclidean/HarMax geometry selects related concepts from the bounded pool. Remaining resolution concerns symbol and router binding. Recommended notation: \(z_i=J_hh_i\) is the live query in the frozen layer-2 JTD frame; a candidate is \(c_{iq}=d_q\) for a layer-2 definition, \(c_{iq}=J_8e_q^{(8)}\), \(c_{iq}=J_{24}e_q^{(24)}\), or a branch/LNGram point already stored in that frame. Use \(d_{iq}^2=\lVert z_i-c_{iq}\rVert_2^2+\lambda_p\delta_{iq}^2+\epsilon^2\), HarMax mass \(p_{iq}\), and \(-\nabla_h\rho(J_hh)=nJ_h^\top\sum_q(y_{iq}-p_{iq})(c_{iq}-z_i)/d_{iq}^2\). Rename LNGram pre-binary channels \(\ell_t=\operatorname{RMSNorm}(h_t)W_q\) and hash addresses \(\eta\). Replace Section 10.7's residual matrix-weighted dot-product router with ordinary Euclidean/HarMax mass over populated skill anchors if the user confirms this binding. D-070 governs the precise scale-invariance claim. |
| O-018 | Concept-cone adjacency capacity and construction policy | Under locked D-071, compare configurable fanout values such as 8, 16, 32, and 64; directed versus mutual adjacency; and fixed-fanout versus distance-radius pruning. Record semantic retrieval recall, causal validity, task accuracy, contraction-pool size, latency, and memory. Precomputed construction distances remain diagnostic metadata while live query-to-anchor distances determine HarMax mass. |
| O-019 | LNGram definition-address assignment and hard-router objective | Compare teacher assignment from verified surface/JTD dictionary-sense seeds, trajectory co-activation, contrastive route learning, and joint HarMax evidence supervision. The present one-bit counterfactual backward pass derives routing gradients from differences between learned continuous table values; converting production payloads to discrete sense handles therefore requires either a supervised route target, a continuous training-only proxy, or an adapted differentiable gather objective. Require stable address-to-sense provenance, exact proxy/handle agreement where applicable, synonym/paraphrase route agreement, definition-sense recall, contextual polysemy resolution, utilization/entropy, collision and dead-route rates, and stability across recurrent rounds and meaning-preserving perturbations. Select the objective from measured retrieval and task evidence before claiming tokenizer-independent semantic equivalence. |
| O-021 | Runtime target-evidence construction for the definition HarMax pool | Specify how surface exact senses, LNGram seed handles, definition-graph neighbors, competing polysemous senses, and contrast candidates populate `evidence`, `supported`, and `valid`. The rule must supply at least one positive supported anchor, preserve zero-target competitors in the HarMax denominator, expose source provenance, and define how route confidence becomes evidence mass before the centroid field is replaced by the signed \(y-p\) operator. |
| O-022 | LNGram threshold stability and multi-route definition aggregation | Use the existing 512 learned four-bit routes as the first redundancy mechanism. Compare calibrated route voting/consensus and confidence weighting under one global top-\(Q\) definition-candidate budget; projection-margin regularization paired with bit/route balance objectives; and a strictly bounded one-bit multiprobe only for logits satisfying \(|z_{t,r,c}|<\tau\). Evaluate the paper's optional multi-table independent projections only if this baseline misses the acceptance target. Measure per-bit absolute margins, Hamming and address churn under meaning-preserving perturbations and across recurrent rounds, related-pair route agreement, unrelated-pair alias rate, bucket occupancy/entropy, dead routes, contextual definition-sense recall@\(Q\), final pool size, task effect, memory traffic, and latency. Lock a policy only from real definition-bank and trajectory evidence. |

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
- SVD-labeled populated skill anchors
- Block-specific shared LoRA factors and per-skill coefficients
- Per-skill/per-block private LoRA residuals
- Skill-to-expert adjacency
- Knowledge/task/skill expert junctions
- Optional routing/skill priors stored separately from expert identity
- Expert-owned reasoning-operator branches and somas
- Appendable experience Engrams
- Two physical blocks reused recurrently
- Block 1 expansion and Block 2 contraction
- Two sequential residual sublayers per physical block
- DeepLoop stability
- Softmax-free vocabulary geometry

---

## 20. Immediate Next Move

### 20.1 Current coding-agent handoff delta

**Asset-transfer checkpoint:** the user is currently downloading the retained
table artifacts from Modal into the local project. Retain the Modal copies
through local verification. When transfer completes, record each local asset
root and validate its manifest, recursive file inventory, total byte count,
row count, shard count, tensor shape, dtype, layer identity, and available
cryptographic hashes before placing those paths into runtime configuration.
The exact downloaded table set and local roots enter this master when the
transfer finishes.

**Current word-definition neighborhood handoff (D-074 through D-086):** the
coding agent's current tree is the implementation authority. Default source
file creation is zero. The preceding v0.8 snapshot supplies historical
evidence and supplies zero filenames for implementation planning.

The coding agent first locates the existing owners of the following contracts,
then returns the exact current paths and current behavior before editing:

1. The existing JTD keeps layer-2 single-word definition senses as the common
   semantic frame and aligns the live state into that frame.
2. The existing complete dictionary bank and its ordered definition-word graph
   supply the semantic neighborhood. Layer-8/layer-24 two-word and three-word
   Engrams retain their separate phrase-memory path and stay outside this
   definition neighborhood.
3. Every complete input or decoded-reasoning word keeps access to its
   single-word sense rows even when phrase memory also fires.
4. LNGram consumes contextual live-state evidence and resolves single-word
   definition senses. Its payload is definition memory; skill routing and
   expert adjacency retain their independent procedural path.
5. HarMax evaluates the bounded set of relevant single-word sense anchors in
   the JTD frame. Context chooses among senses and nearby definition concepts;
   integer token-ID values remain lookup/provenance labels.

**Current-tree inspection and implementation reported this turn:**

1. `joint_transfer.py` already expresses the intended ownership: JTD maps
   definition, phrase, and live continuous views; LNGram pulls from the mapped
   definition space.
2. Coding-agent report: `expert_graph.py` now gathers candidates only through
   `skill_to_experts`; `concept_ids` remain ranking metadata. Two regression
   tests pass, the focused file reports 4/4, and the full suite reports
   136 passed and zero failed. Call-site inspection found
   `ExpertGraph.route()` used only by tests, while live execution uses
   `SkillExpertSystem` and `ComposedWorkingLoRA`; this is contract cleanup
   rather than a live-routing behavior change.
3. `lngram.py` currently implements the paper-native learned `nn.Parameter`
   value tables and projects those learned values back into the live width.
   D-077/D-083 instead require the exact address to pull one or more dictionary
   sense handles, which then gather frozen layer-2 vectors from the runtime
   bank tensor. Address arithmetic and dictionary-row arithmetic remain
   separate.
4. `memory_fusion._definition_field` currently computes inverse-distance mass
   and centroid movement. That establishes attraction but lacks explicit target
   mass (y), signed coefficients (y-p), and repulsion. The full operator
   already exists conceptually and in `geometric_attention.contract_pool`; the
   definition path must reuse or exactly match that operator.
5. `retrieval.py` and `memory_pipeline.py` expose lower-order decomposition
   candidates. Acceptance requires an end-to-end trace proving that every
   constituent dictionary sense reaches the recurrent `MemoryPayloads`
   definition tensor when phrase memory also fires.
6. The available v0.8 snapshot contains `hash_memory_width = 88` in the
   production configuration and `28` in the tiny smoke configuration. This
   width belongs to the trainable Hash-Engram value path. The coding agent's
   current-tree report remains the authority for the active value.
7. The current one-bit counterfactual LNGram backward pass obtains gradients
   from continuous differences between neighboring learned value-table rows.
   Replacing those production values with integer sense handles removes that
   gradient source unless O-019 selects a supervised route objective, a
   training-only continuous proxy, or an adapted differentiable gather.
8. `FrozenDefinitionStore.get()` traverses metadata and sharded storage. It is
   an asset/loading interface, not a per-block recurrent kernel. D-089 requires
   the address-to-sense record and definition-vector gather to be tensorized
   before the live unroll.
9. Coding-agent asset inspection reports 1,532,746 sense rows, 31 safetensors
   shards, approximately 5.8 GiB of 2,048D BF16 layer-2 vectors, and a SQLite
   catalog. These vectors already occupy the canonical JTD definition geometry.
   The available `definition_bank.py` describes schemas and records
   `runtime_storage="cpu_disk_sparse_lookup"`; it does not itself create a
   resident model tensor. The remaining runtime work is bulk materialization
   or mapping plus sparse LNGram row selection.
10. The coding-agent explanation of the prospective HarMax rewrite swapped
    mathematical roles. Under D-092, `y` and `p` are scalar masses, the anchor
    and query are `c` and `u`, and repulsion follows `p > y`. The report also
    labeled `movement_to_live` as an inverse; D-085 preserves it as an
    independently learned map.
11. Coding-agent call-site inspection reports that `DendritronLM.__init__`
    contains token embeddings, position geometry, memory fusion, recurrent
    core, and vocabulary head while exposing no complete definition-bank
    tensor. Present surface preprocessing fetches selected rows and constructs
    `[B,T,S,2048]` payloads before forward. D-093 adds the missing startup bank
    attachment needed for live LNGram-generated addresses.
12. Coding-agent report: `DendritronLM.load_definition_bank`,
    `SparseMemoryFusion.attach_definition_bank`, and the bank-gather branch in
    `_definition_field` now accept supplied sense-row tensors and gather frozen
    rows without recurrent disk access. Eleven new tests reportedly bring the
    invoked suite to 47 passes. This is the materialization half of the bridge:
    the tests begin with `definition_sense_rows`, while the current LNGram
    tables still return learned width-4 values and therefore do not yet produce
    those sense rows.
13. The same report describes the complete approximately 5.85 GiB matrix as a
    persistent buffer registered at the model and propagated to memory fusion.
    PyTorch persistent buffers enter `state_dict()` and participate in module
    device traversal. Before acceptance, enforce D-098: one physical owner,
    checkpoint exclusion, explicit CPU residence/bounded transfer behavior,
    and tests covering `state_dict`, `.to(device)`, shared-storage ownership,
    active out-of-range rows, and exact bank-byte invariance.

**Implementation order:** the bounded expert-candidate contract correction is
reported complete. The D-089 tensorized gather now has a coding-agent-reported
implementation for supplied sense rows, pending the D-098 storage correction
and full-suite reconciliation. Hold the LNGram payload rewrite until two
remaining contracts are explicit: (a) the exact O-019 training signal for
`W_q` after production values become sense handles; and (b) the O-021
construction of target evidence `y`, including positive related anchors and
zero-target competitors. Then implement the smallest existing-file flow:

```text
h_t → J_h → joint query u_t → LNGram W_q address
    → populated bounded sense-row record
    → tensorized frozen-definition gather and adjacency expansion
    → HarMax (y-p) field in joint space
    → joint_to_live → gated residual
```

Phrase layer-8/layer-24 memory and Hash-Engram remain independent branches;
only the definition branch adopts this ordering. A raw latent address selects
an address record; its numerical value never doubles as a definition row index.
The bank attachment uses an external frozen tensor owner or an explicitly
nonpersistent buffer so model checkpoints and generic device moves do not
duplicate or migrate the entire approximately 5.8 GiB asset.

Recovered project history confirms this ordering rather than adding it after
the fact. The completed JTD report closed full anchor export and \(J_8/J_{24}\)
fitting, then named “loading the fitted JTD projections, surface index, token
embeddings, and definition bank into the Dendritron model” as the next
recipient-operator assembly task. That record establishes model attachment as
the pending contract. It leaves full RAM preload versus memory mapping open;
D-093 and O-016 select the physical owner without changing the semantic flow.

After that audit, the coding agent edits the smallest existing integration
surface that lacks one of these contracts. Likely owners include the current
LNGram, definition-memory fusion, retrieval/payload, and recurrent wiring
modules; the agent uses the actual current filenames. Existing implementations
that already satisfy a contract receive tests and documentation updates only.

Acceptance evidence covers: phrase memory and single-word definition memory
coexisting; single-word definition anchors forming the semantic neighborhood;
tree-bark and canine-bark senses occupying different contextual neighborhoods;
token-ID renumbering preserving semantic distances; LNGram payloads containing
definition memory; independent skill/expert routing; frozen definition bytes;
and exact HarMax signed-field equivalence. The returned handoff contains the
actual changed paths, raw test output, and a statement of every preserved
runtime equation.

**Preserve:** the D-048 through D-052 composed-LoRA equation, block/skill
factor ownership, production freeze policy, residual-memory interface, LNGram
live-state provenance through \(W_q(J_hh_t)\), and initial live-state
construction.

**Reported complete in the coding agent's current tree; preserve its current
implementation and evidence:**

- formal causal-provenance tests for D-060 through D-062;
- a small offline verified-transition record builder carrying
  \((h_t,h_{t+1},\Delta h_t,\text{skill},\text{block},\text{round},
  \text{success evidence},\text{ordering})\);
- procedural-delta tests for D-063.

The current-tree audit also reports whether D-066's `(task_id, trajectory_id)`
pair identity and collision test already exist. Any remaining provenance delta
uses the same smallest-existing-surface rule.

**Keep behind evidence or approval gates:** the (z)/JTD/HarMax coordinate and
skill-router reconciliation under O-017, contraction-map ordering under O-015,
Gather-then-Compute kernel selection under O-016, and shared/private fitting
plus offline private-skill commits under O-013 and D-052.

The immediate execution sequence occurs inside the coding agent's current
tree:

1. Report the current HEAD/branch and exact existing owners of JTD definition
   lookup, LNGram definition retrieval, definition-memory fusion, phrase
   memory, and recurrent wiring.
2. Compare their current behavior with D-074 through D-082 and mark each
   contract satisfied or missing.
3. Apply the smallest edit to the existing owner for each measured gap and
   create zero source files by default.
4. Run the complete CPU test suite and preserve its raw output, including:
   - at fixed \(h_t\), adapter routing and \(W_q(h_t)\) remain invariant under
     raw-memory perturbation, while active-gate end-to-end runs preserve the
     allowed mediated path through \(h_t\);
   - zero memory gates remove memory-value and memory-projection influence,
     while injection-boundary outputs match
     \(h+\gamma\odot\Delta m\);
   - donor rows remain outside optimizer groups and byte-identical across
     forward, backward, optimizer step, and recurrent unroll;
   - verified trajectory collection emits
     \(\Delta h_t=h_{t+1}-h_t\), returns zero for stalled states, preserves
     translation invariance, and filters by success evidence.
5. Return exact changed paths, a concise diff summary, raw tests, and every
   preserved runtime equation for master reconciliation.
6. Run O-015 to choose and lock the contraction-sublayer evaluation order when
   the current semantic path is verified.
7. Benchmark O-016 on the current tree and lock the CPU sparse-execution
   kernel from measured equivalence and performance.
8. Lock the O-013 fitter choice, then implement the procedural trajectory,
   shared-fit, private-residual, replay, and versioned-commit pipeline.

The completed punctuation-v2 Stage-1 inventory and corrected Stage-2 donor bank
remain the canonical data baseline. A future recount or migration begins
through an explicit new decision.

---

## 21. Shared/Private Fitter Research Note

Status: **OPEN under O-013; recommended baseline recorded for implementation
comparison.**

For verified target update \(T_{s,b}\), define each block-specific shared atom
and its coupled two-block tuple:

\[
C_{b,j}=b_{b,j}a_{b,j}^{\top},
\qquad
\mathbf C_j=(C_{1,j},C_{2,j}),
\qquad
\sum_b\|C_{b,j}\|_F^2=1
\]

The shared fit is coupled across skills and blocks:

\[
\min_{\{C_{b,j}\},\{q_s\}}
\sum_{s,b}
\left\|
T_{s,b}-\sum_{j=1}^{K}q_{s,j}C_{b,j}
\right\|_F^2
\]

with the same \(q_s\) used in both DeepLoop blocks. Normalize shared atoms,
choose a deterministic sign convention, and whiten or otherwise gauge-fix the
coefficient columns. These constraints resolve scale, sign, and permutation
freedoms sufficiently for stable labels and comparisons.

After freezing the shared fit, compute the coupled-span residual:

\[
R_{s,b}
=
T_{s,b}-\sum_{j=1}^{K}q_{s,j}C_{b,j}
\]

where the \(q_s\) values come from projection onto the coupled span
\(\operatorname{span}\{\mathbf C_1,\ldots,\mathbf C_K\}\) under the inner
product \(\langle\mathbf X,\mathbf Y\rangle=\sum_b\langle X_b,Y_b\rangle_F\).
Initialize the private adapter from a balanced truncated SVD of \(R_{s,b}\):

\[
R_{s,b}\approx U_r\Sigma_rV_r^\top,
\qquad
B_{s,b}^{\mathrm{priv}}=U_r\Sigma_r^{1/2},
\qquad
A_{s,b}^{\mathrm{priv}}=\Sigma_r^{1/2}V_r^\top
\]

During private refinement, preserve separation through projected gradients or
an overlap penalty:

\[
\sum_j
\left\langle
B_{s,b}^{\mathrm{priv}}A_{s,b}^{\mathrm{priv}},
C_{b,j}
\right\rangle_F^2
\]

and optionally add a product-level nuclear/Frobenius budget for private energy.
This staged orthogonal-residual method is the recommended baseline. A
constrained joint optimizer remains an ablation after the staged model is
stable.

Two precision notes govern the experiment:

1. A private-only zero-error solution exists whenever private capacity reaches
   the target update rank. With provisional rank four, private factors can
   still capture dominant low-rank energy and starve shared usage, so the
   identifiability concern remains material while the “zero error” outcome
   depends on target rank.
2. Share initializes principal bases by applying SVD to stacked LoRA \(A\) and
   \(B\) factor vectors separately. A principal component of vectorized
   \(\Delta W\), reshaped into a matrix, may itself have matrix rank above one.
   A forced rank-one refactor therefore adds approximation error. The
   Dendritron fitter must compare factor-SVD, coupled HOSVD/CP, and any
   vectorized-update initialization under the same reconstruction and task
   metrics.

---

## 22. Conversation-to-Master Update Protocol

This same file remains the authoritative running copy for every Dendritron
turn.

1. Read the current master and the relevant current implementation, diff, and
   test evidence when available.
2. State the relevant existing design before evaluating a new proposal.
3. Give the technical answer, disagreement, concern, or missing proof.
4. Apply code changes only when the turn authorizes implementation and the
   target worktree is available.
5. Write every turn back into this master:
   - settled architecture becomes a LOCKED decision and normative text;
   - superseded architecture remains visible as REVOKED history;
   - unresolved hypotheses become OPEN decisions or research notes;
   - implementation and tests enter the project/evidence ledger with their
     provenance;
   - user-reported runs remain labeled user-reported until independently
     reproduced.
6. Preserve this Library item identity and its version history.

Every coding-agent handoff and returned-diff review begins with this compact
change-control header:

- **Classification:** Architecture/Intent, Implementation/Verification, or
  Asset/Configuration.
- **Changed:** exact equations, files, data contracts, or paths affected.
- **Preserved:** applicable LOCKED decisions and user-intent invariants.
- **Evidence status:** proposed, coding-agent-reported, synchronized, locally
  reproduced, or measured on real assets.
- **Open risk:** remaining mathematical, causal, data, or systems uncertainty.
- **Approval required:** yes for Architecture/Intent; stated explicitly for
  every other consequential choice.

The current intent checksum remains:

1. Frozen donor memory and live recurrent compute remain separate.
2. The evolving DeepLoop state carries the whole problem and intermediate
   reasoning.
3. JTD keeps frozen layer-2 definition geometry as the common concept frame and
   aligns layer-8, layer-24, and live-state views into it.
4. LNGram transforms the evolving live state into exact discrete latent
   n-gram addresses whose entire payload is dictionary-sense/word-definition
   memory and bounded definition adjacency; addressed memory returns through
   gated residual paths. JTD and the existing single-word definition bank
   supply the semantic frame and values.
5. Geometric attention uses ordinary Euclidean distance, HarMax inverse-distance
   mass, Harmonic Loss residual, and its negative derivative to produce signed
   attraction and repulsion over causally valid evidence.
6. Procedural SVD labels or proposes skill slots from successful state changes;
   each selected skill still receives the complete live state.
7. A complete skill combines reusable shared LoRA structure with its own
   block-specific private LoRA residual.
8. Skills route to high-dimensional experts through an independent
   skill-to-expert graph; typed branches execute and verify operations, and the
   two recurrent blocks carry thought across rounds.
9. Runtime parameters stay frozen during task execution; learning enters
   through verified offline fitting, replay, regression checks, and versioned
   commit.

### 22.1 Turn ledger

| Date | Turn update | Classification |
|---|---|---|
| 2026-08-15 | Re-established mandatory every-turn master writeback; separated frozen donor memory from live DeepLoop state; locked LNGram as live-state addressing with gated residual retrieval; replaced principal-direction-only skills with complete composed shared/private per-skill LoRAs; separated semantic and procedural SVD; recorded the 73-test checkpoint and required wiring tests. | Locked decisions D-043 through D-057 |
| 2026-08-15 | Reviewed the coding agent's identifiability/non-convexity critique. Recorded shared-first orthogonal-residual fitting as the preferred baseline, corrected the private-only zero-error condition, added gauge fixing and cross-shape SVD caveats, and required a current-code/master sync before trajectory-pipeline work. | Open decisions O-013 and O-014; Section 21 research note |
| 2026-08-15 | Audited the coding agent's 77-test alignment report against the source snapshot available in this workspace. Recorded the newer tree as coding-agent-reported evidence; independently ran the preceding snapshot's 72 discovered tests with 58 passes and 14 environment-gated skips; restored completed punctuation-v2 and donor migration status; classified procedural SVD, fitter, and offline private learning as pending implementation; opened an explicit nonlinear evaluation-order ablation. | Locked decisions D-056, D-058, D-059; open decisions O-014 and O-015 |
| 2026-08-15 | Refined the proposed runtime assertions into causal acceptance tests: fixed-live-state dependency isolation plus permitted mediated memory influence, residual-boundary equivalence, optimizer and byte-level donor invariance, and verified delta-record contracts for procedural SVD. Kept the D-052 offline lifecycle pending and placed Gather-then-Compute behind exact-tree equivalence and CPU benchmarks. | Locked decisions D-060 through D-063; open decision O-016 |
| 2026-08-15 | Converted the alignment audit into a self-contained coding-agent handoff: preserve the current runtime equations, add the formal provenance suite, implement the verified-transition record builder required by procedural SVD, and hold ordering, kernel, and fitter choices behind their evidence gates. | Locked decision D-064; Section 20.1 current implementation delta |
| 2026-08-15 | Recorded active recovery of the retained table artifacts from Modal into the local project. Local wiring begins after manifest, inventory, byte-count, row-count, shard, shape, dtype, layer, and hash verification; exact local roots and table identities will be appended when the transfer completes. | User-reported transfer in progress; Section 20.1 asset-transfer checkpoint |
| 2026-08-15 | Recorded the coding agent's three-file transition/provenance implementation and reported 101-pass CPU run. The 24 new tests cover D-060 through D-063 while preserving the runtime equations. Tightened the collector contract so initial discovery carries task/trajectory identity, structured success evidence, per-trajectory ordering, optional observed skill IDs, immutable detached snapshots, and a delta-only canonical SVD path. | Coding-agent-reported implementation pending local reproduction; locked decision D-065; Stage 8P and Section 20.1 updated |
| 2026-08-15 | Recorded commit `9ab4c8634f1e10bb01103f46c228b1a0c4814227`, its reported 120/120 run, the completed D-065 fields and tests, and the local 248,077-entry Qwen-tokenizer asset. Reconciled the test total as 77 runtime + 12 provenance + 12 D-063 + 19 D-065. Identified the remaining cross-task collision: trajectory ordering and exact identity use `(task_id, trajectory_id)`, with explicit provenance required by the canonical SVD path. | Coding-agent-reported commit pending synchronization and local reproduction; locked decision D-066; Stages 4A, 8, and 8P updated |
| 2026-08-15 | Added explicit user-intent change control. Future reviews classify architecture, implementation, and asset changes; state exact changes and preserved invariants; expose evidence level and open risk; and obtain an explicit user decision for Architecture/Intent changes. Recorded the seven-item current intent checksum covering memory/compute separation, complete live-state skills, residual retrieval, procedural SVD, composed LoRA, experts/branches, and verified offline learning. | Locked decision D-067; Section 22 change-control protocol |
| 2026-08-15 | Added geometric attention, HarMax/Harmonic Loss geometry, LNGram, and JTD explicitly to the protected intent checksum. Found that \(z\) currently names several different quantities and that Section 10.7 retains a matrix-weighted dot-product skill router despite the Euclidean/HarMax geometry lock. Preserved current code while opening a user-approval decision on reserving \(z=J_hh\) for the JTD-aligned live point, renaming address/candidate variables, differentiating Harmonic Loss through \(J_h\), and using Euclidean/HarMax skill-anchor routing. | Locked decision D-068; open Architecture/Intent decision O-017 |
| 2026-08-15 | Audited the proposed JTD/LNGram/HarMax binding against the attached LNGram and Locality Preserving Joint Transfer papers and the Harmonic Loss formulation. Preserved the HarMax target-minus-mass derivative and corrected four surrounding claims: exact address retrieval supplies a candidate pool for local geometric scoring; JTD maps are learned source-specific alignments whose orthogonal status depends on fitter constraints; live queries and memory anchors require the same JTD frame; and private-update orthogonality operates in parameter-update space while frozen storage protects lexical memory. Also limited exact scale invariance to uniform scaling of the completed distance vector unless auxiliary distance terms are normalized compatibly. | Implementation/mathematical clarification; locked decisions D-069 and D-070; O-017 binding awaits user approval |
| 2026-08-15 | Clarified related-concept access as two coordinated operations: exact addressing determines where and when, while live Euclidean/HarMax geometry determines which related concepts participate and with what attraction or repulsion. Recorded bounded precomputed seed adjacency as the recommended candidate while preserving the expansion mechanism for explicit approval. | Architecture/Intent clarification; D-069 refined; open decision O-018; O-017 remains the symbol and skill-router binding gate |
| 2026-08-15 | User explicitly approved bounded offline Euclidean concept-cone adjacency and requested removal of the earlier neighbor-search terminology. Locked seed-and-expand: exact addresses select seeds, Gather-then-Compute expands and filters their adjacency lists, and the evolving live query recomputes Euclidean/HarMax weights and Harmonic derivative forces. Preserved adjacency fanout and construction thresholds as measured configuration choices. | Architecture/Intent approval; locked decision D-071; O-018 narrowed to capacity and construction policy |
| 2026-08-15 | Audited the proposed “100% correct” HarMax explanation. Preserved its moving-live-state, positive-attraction, and zero-target-mass repulsion interpretation. Corrected the class-center role: concept anchors are the finite centers, while the live state is the transient query moved by their force field. Corrected equilibrium to zero net force, which may occur at an anchor or between several anchors; distinguished this from the fixed point of the complete recurrent block; and recorded local scoring cost as \(O(Q_i d)\) after average constant-time address lookup. | Implementation/mathematical clarification; locked decisions D-072 and D-073; runtime equations preserved |
| 2026-08-15 | Clarified the intended local semantic generalization path and corrected LNGram scope. Exact phrase memory and per-word dictionary seeding coexist: every complete input or decoded-reasoning word emits sense seeds even when a bigram/trigram wins, while latent steps obtain dictionary senses through LNGram. LNGram is confined to word-definition memory and bounded definition adjacency; skill routing and skill-to-expert adjacency remain independent procedural systems, with definition evidence entering them only through the evolved live state or post-adjacency conditioning. Recorded token IDs as lookup labels whose integer values carry no semantic distance, including sense-specific `bark` neighborhoods. Audited the coordinate map: `joint_to_live` is an identity-initialized learned movement map, and verified \(\Delta h\) SVD proposes activation-space skill fingerprints before the separate LoRA fitter. | Architecture/Intent correction; locked decisions D-074 through D-080; open training decision O-019 |
| 2026-08-15 | User clarified that JTD and the completed definition assets already perform the semantic alignment, and that local generalization draws from single-word sense anchors while two-word/three-word Engrams retain their separate phrase role. Revoked the proposed registry/builder layer and the inferred new-file plan. Replaced it with a current-tree-first coding handoff: default file creation zero, coding agent identifies the existing owners, edits the smallest current integration surface, and reports actual paths and tests. | Architecture/Intent correction and implementation-process correction; D-081 revoked; D-082 locked; O-020 closed by existing JTD contract |
| 2026-08-15 | Audited the six-point LNGram/JTD critique together with the coding agent's current-tree report. Locked the division that JTD maps continuous views while LNGram pulls sense handles from the completed definition space; specified the exact address → bounded sense IDs → frozen vectors bridge; limited collision-free language to route-offset table layout; separated the chain-rule transpose \(J_h^\top\) from the independent runtime `joint_to_live` map; distinguished frozen layer-2 landmarks from the evolving live reasoning state; and recorded three implementation gaps: learned LNGram value tables, direct concept-to-expert candidate gathering, and attraction-only definition fusion lacking the full \(y-p\) HarMax field. Set the expert-graph correction first and required an exact existing-file tensor-flow plan before the LNGram payload edit. | Architecture/Intent clarification and implementation audit; D-083 through D-086 locked; Sections 4, 7, and 20.1 updated |
| 2026-08-15 | Clarified the final mixed-verdict audit and incorporated the coding agent's call-site finding. The critique correctly identified sign-code collision limits and the discrete/continuous boundary, while its layer-depth dichotomy, invented-map concern, and 88-hallucination concern conflicted with the documented design and inspected source. Confirmed the substantive integration gaps as paper-native learned LNGram values, attraction-only definition fusion, reversed LNGram-to-definition-pool dataflow, and unproven decomposition-to-payload materialization. The direct concept-to-expert edge remains a contract-cleanup issue in `ExpertGraph`; current runtime instead routes through `SkillExpertSystem` and `ComposedWorkingLoRA`, so that isolated edit and regression test do not alter live routing behavior. | Implementation-verdict clarification; no architecture change; D-083 through D-086 preserved; zero new source files |
| 2026-08-15 | Recorded the coding agent's reported `ExpertGraph` correction and 4/4 focused tests, explicitly classifying it as test-path contract cleanup because live routing uses `SkillExpertSystem`. Audited the complete LNGram-to-definition trace and placed the four-file payload rewrite behind three concrete gates: JTD-aligned input to `W_q`, a replacement/adaptation for the counterfactual routing gradient that currently depends on learned value rows, an explicit inference-time target-evidence rule for the signed HarMax field, and a tensorized definition gather that keeps SQLite/shard work outside recurrent visits. Also preserved phrase Engrams and Hash-Engram as independent memory branches rather than swapping the whole fusion module. | Coding-agent-reported implementation plus mathematical/runtime audit; D-087 through D-089 locked; O-019 refined; O-021 opened |
| 2026-08-15 | Reconciled the user's intended “definitions already live in latent space” model with the physical assets and code. Recorded the reported 1,532,746-row, 31-shard, approximately 5.8 GiB BF16 layer-2 bank as the canonical frozen landmark matrix; classified shards/SQLite as persistence and provenance; and required one startup resident/memory-mapped tensor surface plus bounded LNGram row selection during recurrence. Clarified that JTD placement removes any second semantic mapping while an address-to-row routing index remains necessary for sparse access. Corrected the coding-agent report's HarMax roles: `y`/`p` are scalar masses, `c`/`u` are vectors, repulsion follows `p>y`, and `joint_to_live` is independently learned. Recorded the reported 136/136 full-suite result for the isolated expert-graph correction. | Architecture/implementation reconciliation; D-090 through D-092 locked; LNGram rewrite remains behind O-019/O-021 |
| 2026-08-15 | Converted the coding agent's model-construction finding into the exact missing runtime contract. Current preprocessing materializes only surface-selected definition payloads, while `DendritronLM` exposes no complete bank tensor for addresses generated during recurrent reasoning. Locked one-time model assembly of the complete frozen row-addressable bank, with an external owner or nonpersistent CPU buffer to avoid approximately 5.8 GiB checkpoint duplication and unintended device migration. Preserved the address → bounded sense-row record → `index_select` bridge; a raw LNGram address remains distinct from a dictionary row. | Implementation clarification; D-093 locked; zero new source files required |
| 2026-08-15 | Recovered an earlier full-export and JTD-fit completion record. Updated stale status labels: 248,077×2,048 token embeddings, 32,768 bigram plus 32,768 trigram anchor triples, and real CPU \(J_8/J_{24}\) fitting are complete; recorded fingerprints, hashes, checkpoint path, and losses 21.849→0.372 and 37.923→0.844. The record explicitly schedules fitted projections, surface index, token embeddings, and definition-bank loading as the next model-assembly task. Scoped its “memory bridge built” statement to the donor-view JTD projections and confirmed the missing bank attachment as unfinished historical Step 3 work. | Recovered execution evidence; D-094 locked; Stages 3B, 4F, 5, and 8 status corrected |
| 2026-08-15 | Resolved the sign-binarization and landmark-force questions against the LNGram formulation and current source snapshot. Recorded that production uses one learned 2,048-channel projection partitioned into 512 four-bit routes, rather than one monolithic 2,048-bit address; the resulting redundancy still leaves semantic aliasing and zero-boundary fragmentation to empirical validation. Opened a bounded route-consensus, margin, low-confidence multiprobe, and optional multi-table comparison with recall, churn, occupancy, latency, and task metrics. Separately confirmed that JTD performs coordinate alignment while the current definition branch explicitly injects gated centroid-directed movement; the intended signed \(y-p\) HarMax field extends that movement with evidence-controlled attraction and repulsion. | Mathematical/implementation clarification; D-095 and D-096 locked; O-022 opened; no code change authorized |
| 2026-08-15 | Recovered the original reason for combining LNGram with JTD from the attached papers. LNGram contributes modality-independent continuous-hidden-state discretization for exact conditional lookup; Locality Preserving Joint Transfer contributes domain-specific continuous projections into a shared latent frame. Locked the full boundary as live continuous state → JTD-aligned continuous query → LNGram discrete handle → gathered continuous definition anchors → HarMax joint-space movement → live residual. Corrected Section 7.1 so `W_q` consumes the JTD-aligned live query under D-087. Current assets cover layer-8, layer-24, and recipient live views; a future image path requires its own cross-modally fitted source-to-joint map. | Architecture rationale clarification; D-097 locked; existing intent preserved; implementation alignment still governed by D-087/O-019 |
| 2026-08-15 | Reviewed the coding agent's reported definition-bank attachment and 47-pass run. Classified the implementation as the supplied-sense-row materialization half of the bridge: it establishes bank attachment and `sense_rows` → `index_select`, while LNGram still returns learned width-4 values and supplies no address-to-sense record. Identified a contract conflict in the reported persistent, model-plus-fusion buffer ownership, which would enter checkpoint/device traversal for an approximately 5.85 GiB matrix. Required one external or specially managed nonpersistent CPU owner, checkpoint/device tests, active-row range validation, exact test command, and reconciliation with the earlier 136-test suite before acceptance. | Coding-agent-reported partial implementation; D-098 locked; Stage 4C-D updated; approval paused for storage correction |
| 2026-08-15 | Explained the PyTorch storage concern precisely. A persistent buffer joins `state_dict()`, adding the full approximately 5.85 GiB definition matrix to routine checkpoints; every registered buffer also participates in `Module.to(...)`. Clarified that `persistent=False` solves checkpoint inclusion while CPU residency still requires an external owner or explicit custom transfer handling. Preserved the intended runtime behavior: one CPU bank copy and bounded gathered-row transfer to the live compute device. | Implementation clarification; D-099 locked; D-098 preserved |
