# Dendritron Recurrent Engram

> **Sparse memory. Procedural skills. Recurrent geometric reasoning.**

A CPU/ARM-first architecture built from exact Engram memory, a broad frozen definition bank, composed skill LoRAs, high-dimensional expert coordinates, signed HarMax geometry, LNGram latent lookup, and two recurrent physical blocks.

---

## Contents

1. [Architecture summary](#architecture-summary)
2. [Memory — what is stored](#memory--what-is-stored)
3. [Phrase Engrams — exact lookup tables](#phrase-engrams--exact-lookup-tables)
4. [Definition memory — broad English sense bank](#definition-memory--broad-english-sense-bank)
5. [Memory runtime — exact retrieval + JTD](#memory-runtime--exact-retrieval--jtd)
6. [Skill LoRAs — procedural movement](#skill-loras--procedural-movement)
7. [How the skill LoRAs will be made](#how-the-skill-loras-will-be-made)
8. [Shared / private / subspace geometry](#shared--private--subspace-geometry)
9. [Reasoning engine — live hidden state](#reasoning-engine--live-hidden-state)
10. [Experts + branches](#experts--branches)
11. [LNGram](#lngram)
12. [Harmonic Loss / HarMax](#harmonic-loss--harmax)
13. [DeepLoop](#deeploop)
14. [Research basis + evidence boundary](#research-basis--evidence-boundary)

---

## Architecture summary

Dendritron separates three jobs that are normally mixed inside a deep Transformer:

| Capacity | Role |
|---|---|
| **Memory** | Exact bigram/trigram Engram tables and a frozen one-word definition bank |
| **Skills** | Operational LoRAs that encode reusable procedural movement |
| **Reasoning** | A live 2,048D hidden state repeatedly transformed by experts, branches, HarMax, LNGram, and DeepLoop |

---

## Memory — what is stored

### Phrase memory

Two exact Engram lookup tables store recurring two-word and three-word phrases. A phrase is the key. Its row contains two frozen 2,048D hidden-state representations: one from Qwen Layer 8 and one from Qwen Layer 24.

### Definition memory

A separate one-word dictionary bank stores one frozen 2,048D Layer-2 hidden state for every retained word sense, together with its exact definition, source identity, sense identity, and ordered definition-word links.

> Qwen is not part of runtime. Once built, the memory is the finished frozen tables plus their lookup indexes.

---

## Phrase Engrams — exact lookup tables

| Table | Rows | Exact key | Frozen row payload |
|---|---:|---|---|
| Bigram Engram | 500,000 | exact two-word phrase | Layer 8 `[2048]` BF16 + Layer 24 `[2048]` BF16 |
| Trigram Engram | 500,000 | exact three-word phrase | Layer 8 `[2048]` BF16 + Layer 24 `[2048]` BF16 |

```text
exact phrase inventory
→ offline Qwen pass
→ final non-padding token
→ hidden_states[8]  [2048]
→ hidden_states[24] [2048]
→ freeze exact Engram row
```

Runtime:

```text
exact phrase
→ exact Engram table row
→ {Layer-8 hidden state, Layer-24 hidden state}
```

---

## Definition memory — broad English sense bank

The dictionary was built broadly first. Coverage against the phrase Engrams was checked afterward.

| Source | Role |
|---|---|
| Open English WordNet Plus 2025+ | general English senses and polysemy |
| English Wiktionary / Wiktextract 2026-07-06 | broad lexical, scientific, and mathematical vocabulary |
| MeSH 2026 descriptors | biomedical and life-science terminology |
| MeSH 2026 supplementary concepts | chemicals, diseases, organisms, protocols, and emerging terms |

```text
retained external word/sense identity
+ exact source definition
→ withhold target headword from Qwen input
→ definition text + fixed readout marker
→ Qwen hidden_states[2]
→ freeze Layer-2 [2048] sense row
→ retain identity/provenance externally
```

After the broad dictionary existed, every unique word in the completed bigram/trigram Engram keys was checked against it. The audit verified coverage; it did not define the dictionary.

---

## Memory runtime — exact retrieval + JTD

```text
exact 3-word Engram
else exact 2-word Engram
else all matching 1-word definition senses
```

The Layer-2 definition geometry is the common 2,048D reference frame. Separate learned maps align phrase Layer-8, phrase Layer-24, and the live recipient hidden state with that frame:

$$
J_8(e^{(8)}), \qquad
J_{24}(e^{(24)}), \qquad
J_h(h), \qquad
\text{definition Layer-2} = I
$$

Integer row IDs are lookup handles. JTD is continuous alignment, not addressing.

---

## Skill LoRAs — procedural movement

Skills encode **how reasoning proceeds**, not the final answer. Candidate operational skills include:

`COMPARE` · `RETRIEVE` · `REVERSE` · `DERIVE` · `VERIFY` · `DECOMPOSE` · `SEQUENCE` · `COMPOSE`

The same operation should remain useful when the domain and final answer change.

---

## How the skill LoRAs will be made

The proposed training path uses human-authored reasoning checkpoints as direct supervision.

1. **Teach one operation at a time.** Train the same procedural move across unrelated domains.
2. **Freeze everything except the selected skill.** Base weights, memory, experts, other skills, and shared structure stay fixed.
3. **Train movement, not wording.** Match the skill-induced hidden-state movement to the next teacher-authored reasoning checkpoint.
4. **Use wrong → corrected paths as training pairs.** Failed continuation = negative direction; corrected continuation = positive direction.
5. **Verify before commit.** Save episode deltas, run held-out and replay tests, reject or roll back bad updates.
6. **Discover shared structure only after real skills exist.** Stack successful real `ΔW` updates and analyze recurring covariance with SVD/HOSVD.

$$
d_{\text{target}}=\operatorname{stopgrad}(h_{\text{next}}-h_{\text{current}})
$$

$$
d_{\text{skill}}=h_{\text{with skill}}-h_{\text{without skill}}
$$

$$
L_{\text{dir}}=1-\cos(d_{\text{skill}},d_{\text{target}})
$$

---

## Shared / private / subspace geometry

Successful operational skill LoRAs are collected first. Recurring cross-skill covariance becomes an **oblique shared LoRA**. Each skill keeps its measured private residual.

$$
\Delta W_{s,b}
=
B_b^{sh}\operatorname{diag}(q_s)A_b^{sh}
+
B_{s,b}^{priv}A_{s,b}^{priv}
$$

The retained **16–32 principled directions** are compact coordinates for reusable shared procedural structure. They are not complete skills.

---

## Reasoning engine — live hidden state

Dendritron's working representation is an evolving **2,048D hidden state** \(h_t\). Frozen Engram and definition rows remain stored values. The live state carries current context, intermediate conclusions, procedural movement, and memory effects through repeated visits.

---

## Experts + branches

Experts are **high-dimensional answer-bearing coordinates** formed from verified specialized structure that is not well represented by the compact procedural skill geometry.

During consolidation, the expert span is protected first. The lower-dimensional skill geometry is then adjusted around it using stable QR/SVD/null-space projection rather than literal classical Gram-Schmidt.

| Branch operator | Role |
|---|---|
| Deductive | premises → conclusion |
| Causal | cause / mechanism / effect |
| Contrastive | separate competing candidates |
| Counterfactual | change one condition and compare |
| Abductive | rank explanations for an observation |
| Compositional / analogical | transfer a relation into another domain |

---

## LNGram

LNGram exists for the **continuous recurrent trajectory**.

```text
successive h_t [2048]
→ RMSNorm
→ W_q
→ 512 × 4 hard bits
→ symbols in {0…15}
→ exact latent 2g / 3g address
→ continuous latent-memory value
```

$$
\operatorname{addr}^{(n)}
=
r\,16^n
+
\sum_j
\operatorname{symbol}(t-n+1+j,r)\,16^j
$$

The address is discrete and deterministic for a given bit sequence. The payload is a continuous trainable latent-memory value.

---

## Harmonic Loss / HarMax

For a bounded candidate pool:

$$
D_q=\|u-c_q\|^2+\text{bounded auxiliary terms}
$$

$$
p_q=\frac{D_q^{-n/2}}{\sum_j D_j^{-n/2}}
$$

$$
\rho=-\sum_q y_q\log p_q
$$

$$
\Delta u
=
n\sum_q
(y_q-p_q)
\frac{c_q-u}{D_q}
$$

- **\(y_q > p_q\)**: attraction toward underrepresented support.
- **\(y_q < p_q\)**: repulsion from excess-proximity competitors.

---

## DeepLoop

Dendritron stores **two ordered physical blocks** and reuses them for \(R\) rounds.

$$
H^{(r,1)}
=
\Phi_1(H^{(r,0)},\text{memory},\text{LNGram},\text{routing})
$$

$$
H^{(r+1,0)}
=
\Phi_2(H^{(r,1)},\text{memory},\text{LNGram},\text{routing})
$$

| Block | Bias |
|---|---|
| **Block 1 — Expansion** | retrieve, associate, propose |
| **Block 2 — Contraction** | contrast, verify, integrate |

Stored block depth remains **2** while the sequential unrolled thought depth is **\(2R\)**.

DeepLoop scaling:

$$
K=2,\qquad N=2R,\qquad M=4R
$$

$$
M\kappa_R\left(\frac{\beta}{\alpha}\right)^2=O(1)
$$

$$
\alpha=\sqrt{4R}=2\sqrt R
$$

$$
\beta=\frac{1}{\sqrt{16R}}=\frac{1}{4\sqrt R}
$$

---


## Architecture flow diagrams

### Surface memory

```mermaid
flowchart TD
    A[Complete-word endpoint] --> B{Exact 3-word phrase?}
    B -->|hit| C[Trigram Engram row]
    B -->|miss| D{Exact 2-word phrase?}
    D -->|hit| E[Bigram Engram row]
    D -->|miss| F[All matching one-word definition senses]
    D -->|miss| H[Trainable Hash-Engram local-pattern path]
    C --> G[Pull frozen Layer-8 + Layer-24 hidden states]
    E --> G
    F --> I[Pull frozen Layer-2 sense anchors]
    G --> J[JTD-aligned memory residual]
    I --> J
```

A phrase hit uses the exact phrase Engram at that endpoint. Constituent senses are **not** automatically injected underneath that phrase hit.

### LNGram latent memory

```mermaid
flowchart TD
    A[Successive live hidden states h_t] --> B[RMSNorm]
    B --> C[Learned projection W_q]
    C --> D[512 four-bit route symbols]
    D --> E[Exact latent 2-gram / 3-gram address]
    E --> F[Continuous trainable LNGram memory value]
    F --> G[Gated residual into live hidden state]
```

LNGram addresses its own latent memory from the recurrent hidden-state trajectory. It does not route into the dictionary-sense table.

### Skills, experts, branches, and recurrence

```mermaid
flowchart TD
    A[Live hidden state h_t] --> B[Route operational skill LoRA]
    B --> C[Apply procedural movement]
    C --> D[Bounded expert coordinates relevant to current task]
    D --> E[Instantiate expert-owned typed branches]
    E --> F[Block 1: expand candidate relations]
    F --> G[Block 2: contract / verify against evidence]
    G --> H[Updated live hidden state]
    H -->|next round| A
```

The skill supplies procedural movement. Expert coordinates carry specialized answer structure; branches bind that structure to the current reasoning state during the recurrent visit.


## Research basis + evidence boundary

The memory assets and JTD fits are real completed/reported artifacts. Several training and consolidation pieces — including the human-directed directional skill objective, oblique shared-LoRA extraction, and expert-coordinate formation — remain proposed experiments until measured on real Dendritron trajectories.

---

**Dendritron Recurrent Engram synthesis and integration — Ryan Carson**
