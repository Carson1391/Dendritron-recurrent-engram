# Dendritron v1.0 Tokenizer and Punctuation Lock

Date: 2026-07-31

This lock supersedes the UTF-8 byte frontend in the v0.8 runtime lock.

## One live tokenizer

The first Dendritron recipient uses the tokenizer from
`Qwen/Qwen3.6-35B-A3B` at the exact resolved revision recorded by the Stage-2
donor manifest.

```text
text
-> raw Qwen IDs
-> Dendritron token embeddings
-> two recurrent blocks
-> Qwen-vocabulary token scores
```

Qwen model weights remain offline donor machinery. CPU tokenization uses the
saved tokenizer artifact and introduces no CUDA or NVIDIA runtime dependency.

## Parallel conditional-memory address stream

Raw Qwen IDs remain exact for the language model. A frozen surjective function
`P` creates canonical IDs solely for surface-memory and Hash-Engram addressing:

\[
x_t' = P(x_t)
\]

`P` follows the official Engram demonstration sequence:

```text
decode one Qwen vocabulary item
-> NFKC
-> NFD
-> strip Unicode accents
-> lowercase
-> collapse space/tab/CR/LF runs
-> preserve one whitespace-only canonical key
-> strip outer whitespace
-> assign one canonical integer per distinct result
```

If a vocabulary item decodes through U+FFFD, its raw tokenizer piece becomes
the key. The projection is built once on CPU, fingerprinted, and stored beside
the tokenizer/surface-index manifest.

The Engram paper reports a 23.43% reduction for its own 128k tokenizer. The
Qwen reduction is measured by the compiler and recorded as
`reduction_percent`; 23.43% is a reference value rather than a Qwen constant.

## Punctuation

Punctuation has two simultaneous roles:

1. Raw Qwen punctuation IDs stay in the hidden-state stream and output
   vocabulary.
2. Canonical punctuation IDs stay in Hash-Engram keys, preserving local syntax.

Frozen phrase Engrams and dictionary senses are complete-word memories. Their
matcher uses Qwen offset mappings to align Unicode word spans to token
positions. Any non-whitespace material between adjacent words creates a frozen
word-Engram boundary. Apostrophes and hyphens surrounded by word characters
remain internal to one word.

Consequences:

- `tree bark,` activates the frozen `tree bark` row at the token containing
  the end of `bark`, even when Qwen fuses the comma into that token.
- `tree, bark` leaves that bigram inactive because the comma is between words.
- `camera-based` and `don't` each remain one complete word.
- The comma, period, question mark, quotes, and other punctuation remain
  available to the recurrent hidden states and Hash-Engram path.

Stage-1 counting follows the same boundary rule, preventing cross-sentence or
cross-clause word n-grams in a rebuilt inventory.

## Acceptance conditions

- Raw Qwen IDs and canonical memory IDs remain separate tensors/namespaces.
- The exact tokenizer revision and projection fingerprint appear in manifests.
- The Qwen-specific effective-vocabulary reduction is measured.
- Case, leading-space, accent, and whitespace equivalents merge in memory IDs.
- Distinct punctuation classes remain distinct after canonicalization.
- Offset-aligned tests cover punctuation fused to a word.
- Frozen word Engrams never cross intervening punctuation.
- Hash-Engram continues to receive punctuation-bearing canonical suffixes.
