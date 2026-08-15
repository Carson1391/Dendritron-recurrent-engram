"""Assemble the Dendritron recipient with frozen assets and test the minimal reasoning path.

This script loads every frozen artifact from the corpus volume into a
production-width DendritronLM and runs one complete forward pass:

1. Load the Qwen tokenizer snapshot from the JTD surface index.
2. Open the surface index, dictionary store, and Engram store.
3. Build a SurfaceMemoryAddressor and resolve a test phrase.
4. Materialize MemoryPayloads from the resolved plans (load actual tensors).
5. Construct DendritronLM at production width (2048D, 248,077 vocab).
6. Load frozen token embeddings and fitted JTD projections.
7. Run the forward pass with memory payloads through the two-block core.
8. Verify output shapes, finiteness, and report top-k tokens.

Run:
    modal run modal_assemble_recipient.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_extract_states import CORPUS_VOLUME_NAME, DATA_MOUNT


APP_NAME = "dendritron-recipient-assembly"

# Frozen artifact paths on the corpus volume
JTD_ROOT = "/data/dendritron-stage4-jtd-punctuation-v2"
LATENT_ASSETS_ROOT = f"{JTD_ROOT}/latent-assets"
TOKEN_EMBEDDINGS_PATH = f"{LATENT_ASSETS_ROOT}/frontend/qwen_token_embeddings.safetensors"
JTD_PROJECTIONS_PATH = f"{JTD_ROOT}/jtd-projections.pt"
SURFACE_INDEX_PATH = f"{JTD_ROOT}/surface_index.sqlite3"
TOKENIZER_SNAPSHOT_DIR = f"{JTD_ROOT}/tokenizer_snapshot"
TOKENIZER_CONTRACT_PATH = f"{JTD_ROOT}/tokenizer_contract.json"
DICTIONARY_BANK_ROOT = "/data/dendritron-stage3-dictionary/bank"
STAGE2_ROOT = "/data/dendritron-stage2-punctuation-v2"

# Test phrase: should hit bigram "tree bark" and dictionary senses for "tree" and "bark"
TEST_PHRASE = "tree bark"

app = modal.App(APP_NAME)
corpus_volume = modal.Volume.from_name(CORPUS_VOLUME_NAME, create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "numpy>=2.1,<3",
        "safetensors>=0.5,<1",
        "torch>=2.7,<3",
        "transformers>=5.4,<6",
        "sentencepiece>=0.2,<1",
        "huggingface-hub[hf-xet]>=0.34,<2",
    )
    .env(
        {
            "HF_HOME": "/models/huggingface",
            "HF_HUB_CACHE": "/models/huggingface/hub",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TOKENIZERS_PARALLELISM": "true",
        }
    )
    .add_local_file(
        local_path=Path(__file__).parent / "modal_extract_states.py",
        remote_path="/root/modal_extract_states.py",
    )
    .add_local_dir(
        local_path=Path(__file__).parent / "stage3_jtd",
        remote_path="/root/stage3_jtd",
    )
    .add_local_dir(
        local_path=Path(__file__).parent / "dendritron",
        remote_path="/root/dendritron",
    )
)


def _materialize_payloads(
    plans: tuple[Any, ...],
    engram_store: Any,
    definition_store: Any,
    memory_width: int,
    max_senses: int = 16,
) -> dict[str, Any]:
    """Convert surface plans into MemoryPayloads tensors for batch_size=1.

    For each token position:
    - Donor Engram hit (word_order 2/3): load layer08 + layer24 rows.
    - Dictionary sense hit (word_order 1): load all matching layer02 sense vectors.
    - Hash Engram miss: keep hash addresses as-is.

    Returns a dict suitable for constructing MemoryPayloads.
    """
    import torch

    seq_len = len(plans)
    phrase_layer8 = torch.zeros(1, seq_len, memory_width, dtype=torch.float32)
    phrase_layer24 = torch.zeros(1, seq_len, memory_width, dtype=torch.float32)
    phrase_mask = torch.zeros(1, seq_len, dtype=torch.bool)
    definitions = torch.zeros(1, seq_len, max_senses, memory_width, dtype=torch.float32)
    definition_mask = torch.zeros(1, seq_len, max_senses, dtype=torch.bool)
    definition_sense_rows = torch.full(
        (1, seq_len, max_senses), -1, dtype=torch.long
    )

    # Hash addresses: {order: [B, T, heads]} with -1 for unavailable
    hash_orders = (2, 3)
    hash_heads = 4
    hash_addresses = {
        order: torch.full(
            (1, seq_len, hash_heads), -1, dtype=torch.long
        )
        for order in hash_orders
    }

    donor_hits = 0
    dict_hits = 0
    hash_misses = 0

    for pos, plan in enumerate(plans):
        # Check for exact memory hits
        exact_memories = plan.exact_memories
        has_donor = False
        has_dict = False

        for memory in exact_memories:
            if memory.word_order in (2, 3):
                # Donor Engram hit: load frozen layer08 + layer24
                payload = engram_store.get(
                    word_order=memory.word_order,
                    row_index=memory.selected[0].row_index,
                )
                phrase_layer8[0, pos] = payload.layer08.to(torch.float32)
                phrase_layer24[0, pos] = payload.layer24.to(torch.float32)
                phrase_mask[0, pos] = True
                has_donor = True
                donor_hits += 1
            elif memory.word_order == 1:
                # Dictionary sense hit: load all matching senses
                senses = memory.selected
                for s_idx, candidate in enumerate(senses):
                    if s_idx >= max_senses:
                        break
                    payload = definition_store.get(candidate.row_index)
                    definitions[0, pos, s_idx] = payload.layer02.to(torch.float32)
                    definition_mask[0, pos, s_idx] = True
                    definition_sense_rows[0, pos, s_idx] = candidate.row_index
                has_dict = True
                dict_hits += 1

        # Hash Engram fallback for positions without a donor hit
        if not has_donor and plan.hash_engram is not None:
            for order in hash_orders:
                addrs = plan.hash_engram.by_order.get(order)
                if addrs is not None:
                    # addrs is a tuple of head values for this position
                    for h_idx, head_val in enumerate(addrs[:hash_heads]):
                        hash_addresses[order][0, pos, h_idx] = int(head_val)
            hash_misses += 1

    return {
        "phrase_layer8": phrase_layer8,
        "phrase_layer24": phrase_layer24,
        "phrase_mask": phrase_mask,
        "definitions": definitions,
        "definition_mask": definition_mask,
        "definition_sense_rows": definition_sense_rows,
        "hash_addresses": hash_addresses,
        "stats": {
            "donor_hits": donor_hits,
            "dict_hits": dict_hits,
            "hash_misses": hash_misses,
            "total_positions": seq_len,
        },
    }


@app.function(
    image=image,
    cpu=8.0,
    memory=65_536,
    timeout=6 * 60 * 60,
    volumes={str(DATA_MOUNT): corpus_volume},
)
def assemble_and_test() -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    from dendritron.config import DendritronConfig
    from dendritron.model import DendritronLM
    from dendritron.memory_fusion import MemoryPayloads
    from dendritron.memory_pipeline import SurfaceMemoryAddressor
    from dendritron.jtd import SurfaceMemoryIndex
    from dendritron.definition_store import FrozenDefinitionStore
    from dendritron.engram_store import FrozenEngramStore
    from dendritron.hash_engram import HashEngramAddressor

    # --- 1. Load tokenizer from snapshot ---
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_SNAPSHOT_DIR,
        trust_remote_code=False,
        use_fast=True,
    )
    vocab_size = len(tokenizer)

    # --- 2. Open frozen stores ---
    surface_index = SurfaceMemoryIndex(
        Path(SURFACE_INDEX_PATH),
    )
    definition_store = FrozenDefinitionStore(
        Path(DICTIONARY_BANK_ROOT),
    )
    engram_store = FrozenEngramStore(
        Path(STAGE2_ROOT),
        maximum_cached_shards=4,
    )

    # --- 3. Build surface memory addressor ---
    # Hash addressor must match model config's hash_table_rows
    hash_engram = HashEngramAddressor(
        orders=(2, 3),
        heads=4,
        table_rows={2: 65_536, 3: 262_144},
    )
    addressor = SurfaceMemoryAddressor(
        exact_index=surface_index,
        hash_engram=hash_engram,
    )

    # --- 4. Tokenize test phrase and resolve memory plans ---
    input_ids = tokenizer.encode(TEST_PHRASE, add_special_tokens=False)
    plans = addressor.resolve_text(tokenizer, TEST_PHRASE)

    # Report resolution results
    resolution_report = []
    for pos, plan in enumerate(plans):
        entry = {
            "position": pos,
            "token_id": int(input_ids[pos]) if pos < len(input_ids) else None,
            "token_text": tokenizer.decode([input_ids[pos]]) if pos < len(input_ids) else None,
            "has_donor_hit": plan.has_frozen_donor_hit,
            "has_dict_candidates": plan.has_dictionary_candidates,
            "has_hash_engram": plan.hash_engram is not None,
        }
        if plan.exact_memory is not None:
            entry["word_order"] = plan.exact_memory.word_order
            entry["selected_count"] = len(plan.exact_memory.selected)
            entry["surface_text"] = plan.exact_memory.selected[0].surface_text
        resolution_report.append(entry)

    # --- 5. Materialize MemoryPayloads ---
    materialized = _materialize_payloads(
        plans,
        engram_store=engram_store,
        definition_store=definition_store,
        memory_width=2048,
        max_senses=16,
    )
    payloads = MemoryPayloads(
        phrase_layer8=materialized["phrase_layer8"],
        phrase_layer24=materialized["phrase_layer24"],
        phrase_mask=materialized["phrase_mask"],
        definitions=materialized["definitions"],
        definition_mask=materialized["definition_mask"],
        definition_sense_rows=materialized["definition_sense_rows"],
        hash_addresses=materialized["hash_addresses"],
    )

    # --- 6. Construct production-width DendritronLM ---
    # Smoke mode for structural testing:
    # - init_mode="smoke" (random init for structural validation)
    # - expert_branches=1 (one typed deductive branch)
    # - use_lngram=False (LNGram connected in step 7a)
    # - loop_rounds=1 (minimal: one pass through both blocks)
    config = DendritronConfig(
        vocab_size=vocab_size,
        model_width=2048,
        memory_width=2048,
        max_sequence_length=256,
        loop_rounds=1,
        context_window=64,
        context_top_k=16,
        max_skill_slots=32,
        shared_basis_count=16,
        max_private_lora_rank=4,
        skill_top_k=4,
        init_mode="smoke",
        expert_count=11,
        expert_hidden_width=512,
        expert_branches=1,
        expert_top_k=2,
        use_lngram=False,
        hash_orders=(2, 3),
        hash_heads=4,
        hash_table_rows=(65_536, 262_144),
        hash_memory_width=88,
    )
    model = DendritronLM(config)

    # --- 7. Load frozen token embeddings ---
    model.load_token_embedding_artifact(TOKEN_EMBEDDINGS_PATH)

    # --- 8. Load fitted JTD projections ---
    model.load_joint_transfer_checkpoint(JTD_PROJECTIONS_PATH)

    # --- 9. Run forward pass ---
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        output = model(
            input_tensor,
            memory_payloads=payloads,
            return_stats=True,
        )

    # --- 10. Verify and report ---
    logits = output.logits
    hidden = output.hidden
    stats = output.recurrent_stats

    # Check finiteness
    logits_finite = bool(torch.isfinite(logits).all())
    hidden_finite = bool(torch.isfinite(hidden).all())

    # Top-k tokens at each position
    top_k = 10
    topk_results = []
    for pos in range(logits.shape[1]):
        topk_vals, topk_ids = logits[0, pos].topk(top_k)
        topk_tokens = [tokenizer.decode([int(tid)]) for tid in topk_ids]
        topk_results.append({
            "position": pos,
            "input_token": tokenizer.decode([int(input_ids[pos])]),
            "top_k_tokens": topk_tokens,
            "top_k_logit_values": [round(float(v), 4) for v in topk_vals],
        })

    # Memory fusion stats from first block visit
    first_visit = stats.visits[0] if stats.visits else None
    memory_stats = None
    if first_visit is not None:
        ms = first_visit.memory
        memory_stats = {
            "block_index": first_visit.block_index,
            "phrase8_gate": float(ms.phrase8_gate),
            "phrase24_gate": float(ms.phrase24_gate),
            "definition_gate": float(ms.definition_gate),
            "hash_gate": float(ms.hash_gate),
            "active_definition_count": (
                None if ms.active_definition_count is None
                else [int(x) for x in ms.active_definition_count[0].tolist()]
            ),
        }

    # Skill-expert stats from first visit
    skill_stats = None
    if first_visit is not None:
        se = first_visit.skill_expert
        skill_stats = {
            "skill_gate": float(se.skill_gate),
            "expert_gate": float(se.expert_gate),
            "skill_indices": [int(x) for x in se.skills.indices[0, 0].tolist()],
            "expert_indices": [int(x) for x in se.experts.indices[0, 0].tolist()],
        }

    result = {
        "completed": True,
        "test_phrase": TEST_PHRASE,
        "vocab_size": vocab_size,
        "input_ids": input_ids,
        "input_tokens": [tokenizer.decode([tid]) for tid in input_ids],
        "sequence_length": len(input_ids),
        "model_config": {
            "model_width": config.model_width,
            "memory_width": config.memory_width,
            "loop_rounds": config.loop_rounds,
            "shared_basis_count": config.shared_basis_count,
            "max_private_lora_rank": config.max_private_lora_rank,
            "init_mode": config.init_mode,
            "expert_branches": config.expert_branches,
            "use_lngram": config.use_lngram,
            "max_skill_slots": config.max_skill_slots,
            "expert_count": config.expert_count,
        },
        "memory_resolution": resolution_report,
        "materialization_stats": materialized["stats"],
        "output_verification": {
            "logits_shape": list(logits.shape),
            "hidden_shape": list(hidden.shape),
            "logits_finite": logits_finite,
            "hidden_finite": hidden_finite,
            "rounds_executed": stats.rounds_executed,
            "alpha": stats.alpha,
            "beta": stats.beta,
            "final_relative_change": (
                None if stats.final_relative_change is None
                else [round(float(x), 6) for x in stats.final_relative_change.tolist()]
            ),
        },
        "memory_fusion_stats": memory_stats,
        "skill_expert_stats": skill_stats,
        "top_k_predictions": topk_results,
        "frozen_assets_loaded": {
            "token_embeddings": TOKEN_EMBEDDINGS_PATH,
            "jtd_projections": JTD_PROJECTIONS_PATH,
            "surface_index": SURFACE_INDEX_PATH,
            "dictionary_bank": DICTIONARY_BANK_ROOT,
            "engram_bank": STAGE2_ROOT,
        },
    }

    # Clean up
    surface_index.close()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


@app.local_entrypoint()
def main() -> None:
    result = assemble_and_test.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
