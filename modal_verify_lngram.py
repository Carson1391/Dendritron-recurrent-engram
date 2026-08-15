"""Verify the already-wired production LNGram path with a 3+ token forward/backward test.

LNGram is already connected in both recurrent blocks:
    HarMax -> JTD/memory fusion -> LNGram -> skill/expert system

This script:
1. Enables use_lngram=True in the production config.
2. Runs a forward pass with a 5+ token input so both order-2 and order-3 activate.
3. Reports each block's LNGram symbol/address shapes and validity.
4. Confirms order-2 and order-3 addresses stay within their table bounds.
5. Runs one backward pass and confirms finite, nonzero gradients for:
   - address_projection.weight (both blocks)
   - tables["2"] (both blocks)
   - tables["3"] (both blocks)
6. Preserves the separate Hash-Engram and LNGram namespaces.

Run:
    modal run modal_verify_lngram.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_extract_states import CORPUS_VOLUME_NAME, DATA_MOUNT


APP_NAME = "dendritron-lngram-verification"

JTD_ROOT = "/data/dendritron-stage4-jtd-punctuation-v2"
LATENT_ASSETS_ROOT = f"{JTD_ROOT}/latent-assets"
TOKEN_EMBEDDINGS_PATH = f"{LATENT_ASSETS_ROOT}/frontend/qwen_token_embeddings.safetensors"
JTD_PROJECTIONS_PATH = f"{JTD_ROOT}/jtd-projections.pt"
SURFACE_INDEX_PATH = f"{JTD_ROOT}/surface_index.sqlite3"
TOKENIZER_SNAPSHOT_DIR = f"{JTD_ROOT}/tokenizer_snapshot"
DICTIONARY_BANK_ROOT = "/data/dendritron-stage3-dictionary/bank"
STAGE2_ROOT = "/data/dendritron-stage2-punctuation-v2"

# 5+ token phrase: ensures order-2 activates at positions >=1 and order-3 at positions >=2
TEST_PHRASE = "the old tree bark is rough"

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
    """Convert surface plans into MemoryPayloads tensors for batch_size=1."""
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
        exact_memories = plan.exact_memories
        has_donor = False

        for memory in exact_memories:
            if memory.word_order in (2, 3):
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
                senses = memory.selected
                for s_idx, candidate in enumerate(senses):
                    if s_idx >= max_senses:
                        break
                    payload = definition_store.get(candidate.row_index)
                    definitions[0, pos, s_idx] = payload.layer02.to(torch.float32)
                    definition_mask[0, pos, s_idx] = True
                    definition_sense_rows[0, pos, s_idx] = candidate.row_index
                dict_hits += 1

        if not has_donor and plan.hash_engram is not None:
            for order in hash_orders:
                addrs = plan.hash_engram.by_order.get(order)
                if addrs is not None:
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
def verify_lngram() -> dict[str, Any]:
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

    # --- 1. Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_SNAPSHOT_DIR,
        trust_remote_code=False,
        use_fast=True,
    )
    vocab_size = len(tokenizer)

    # --- 2. Open frozen stores ---
    surface_index = SurfaceMemoryIndex(Path(SURFACE_INDEX_PATH))
    definition_store = FrozenDefinitionStore(Path(DICTIONARY_BANK_ROOT))
    engram_store = FrozenEngramStore(Path(STAGE2_ROOT), maximum_cached_shards=4)

    # --- 3. Build surface memory addressor ---
    # Hash-Engram: token-ID fallback, separate namespace from LNGram
    hash_engram = HashEngramAddressor(
        orders=(2, 3),
        heads=4,
        table_rows={2: 65_536, 3: 262_144},
    )
    addressor = SurfaceMemoryAddressor(
        exact_index=surface_index,
        hash_engram=hash_engram,
    )

    # --- 4. Tokenize test phrase ---
    input_ids = tokenizer.encode(TEST_PHRASE, add_special_tokens=False)
    assert len(input_ids) >= 3, (
        f"Test phrase must produce >=3 tokens for LNGram order-3 activation, "
        f"got {len(input_ids)}: {input_ids}"
    )
    plans = addressor.resolve_text(tokenizer, TEST_PHRASE)

    # --- 5. Materialize MemoryPayloads (Hash-Engram namespace) ---
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

    # --- 6. Construct production DendritronLM with LNGram ENABLED ---
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
        use_lngram=True,          # LNGram enabled in both blocks
        lngram_bits_per_route=4,
        lngram_orders=(2, 3),
        lngram_route_memory_width=4,
        hash_orders=(2, 3),
        hash_heads=4,
        hash_table_rows=(65_536, 262_144),
        hash_memory_width=88,
    )
    model = DendritronLM(config)

    # --- 7. Load frozen assets ---
    model.load_token_embedding_artifact(TOKEN_EMBEDDINGS_PATH)
    model.load_joint_transfer_checkpoint(JTD_PROJECTIONS_PATH)

    # --- 8. Report LNGram table sizes ---
    lngram = model.core.lngram
    assert lngram is not None, "LNGram must be enabled"
    table_row_counts = lngram[0].table_row_counts
    route_count = lngram[0].route_count
    alphabet_size = lngram[0].alphabet_size
    bits_per_route = lngram[0].bits_per_route

    # --- 9. Forward pass with stats ---
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    output = model(input_tensor, memory_payloads=payloads, return_stats=True)

    logits = output.logits
    hidden = output.hidden
    stats = output.recurrent_stats

    # --- 10. Extract per-block LNGram statistics ---
    block_lngram_reports = []
    for visit in stats.visits:
        block_idx = visit.block_index
        lngram_stats = visit.lngram
        if lngram_stats is None:
            block_lngram_reports.append({
                "block_index": block_idx,
                "lngram_active": False,
            })
            continue

        symbols = lngram_stats.symbols
        # symbols shape: [B, T, routes]
        report = {
            "block_index": block_idx,
            "lngram_active": True,
            "symbols_shape": list(symbols.shape),
            "route_count": int(symbols.shape[-1]),
            "orders": {},
        }

        for order in lngram_stats.addresses:
            addresses = lngram_stats.addresses[order]
            valid = lngram_stats.valid[order]
            gates = lngram_stats.gates[order]
            table_rows = table_row_counts[order]

            # Check all addresses are within table bounds
            valid_addrs = addresses[valid]
            max_addr = int(valid_addrs.max().item()) if valid_addrs.numel() > 0 else -1
            min_addr = int(valid_addrs.min().item()) if valid_addrs.numel() > 0 else -1
            in_bounds = bool((valid_addrs < table_rows).all()) if valid_addrs.numel() > 0 else True
            in_bounds = in_bounds and bool((valid_addrs >= 0).all()) if valid_addrs.numel() > 0 else in_bounds

            # Count valid positions per batch element
            valid_per_position = valid.any(dim=-1)  # [B, T]
            valid_count = int(valid_per_position.sum().item())

            report["orders"][str(order)] = {
                "addresses_shape": list(addresses.shape),
                "valid_shape": list(valid.shape),
                "gates_shape": list(gates.shape),
                "table_rows": int(table_rows),
                "min_address": min_addr,
                "max_address": max_addr,
                "in_bounds": in_bounds,
                "valid_positions": valid_count,
                "total_positions": int(addresses.shape[1]),
                "gates_mean": float(gates.mean().item()),
                "gates_std": float(gates.std().item()),
            }

        block_lngram_reports.append(report)

    # --- 11. Backward pass ---
    # Use a simple loss to ensure gradients flow through LNGram
    loss = logits.square().mean()
    loss.backward()

    # --- 12. Check LNGram gradients ---
    gradient_report = {
        "loss_value": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss)),
    }

    all_gradients_ok = True
    for block_idx in range(2):
        block_key = f"block_{block_idx}"
        gradient_report[block_key] = {}

        # address_projection gradient
        ap_grad = lngram[block_idx].address_projection.weight.grad
        if ap_grad is not None:
            ap_finite = bool(torch.isfinite(ap_grad).all())
            ap_nonzero = bool(ap_grad.abs().sum() > 0)
            ap_norm = float(ap_grad.norm().item())
        else:
            ap_finite = False
            ap_nonzero = False
            ap_norm = 0.0

        gradient_report[block_key]["address_projection"] = {
            "gradient_norm": ap_norm,
            "finite": ap_finite,
            "nonzero": ap_nonzero,
        }
        if not (ap_finite and ap_nonzero):
            all_gradients_ok = False

        # Table gradients for each order
        for order in (2, 3):
            table_grad = lngram[block_idx].tables[str(order)].grad
            if table_grad is not None:
                tg_finite = bool(torch.isfinite(table_grad).all())
                tg_nonzero = bool(table_grad.abs().sum() > 0)
                tg_norm = float(table_grad.norm().item())
                tg_nonzero_rows = int((table_grad.abs().sum(dim=-1) > 0).sum().item())
            else:
                tg_finite = False
                tg_nonzero = False
                tg_norm = 0.0
                tg_nonzero_rows = 0

            gradient_report[block_key][f"table_order_{order}"] = {
                "gradient_norm": tg_norm,
                "finite": tg_finite,
                "nonzero": tg_nonzero,
                "nonzero_rows": tg_nonzero_rows,
                "total_rows": int(table_grad.shape[0]) if table_grad is not None else 0,
            }
            if not (tg_finite and tg_nonzero):
                all_gradients_ok = False

    gradient_report["all_gradients_ok"] = all_gradients_ok

    # --- 13. Verify Hash-Engram and LNGram namespace separation ---
    # Hash-Engram tables live in model.memory_fusion.hash_memory
    # LNGram tables live in model.core.lngram[block].tables
    hash_table_params = set()
    for order, table in model.memory_fusion.hash_memory.tables.items():
        hash_table_params.add(id(table.weight))

    lngram_table_params = set()
    for block_idx in range(2):
        for order, table in lngram[block_idx].tables.items():
            lngram_table_params.add(id(table))

    namespace_disjoint = hash_table_params.isdisjoint(lngram_table_params)

    # --- 14. Assemble result ---
    result = {
        "completed": True,
        "test_phrase": TEST_PHRASE,
        "input_ids": input_ids,
        "input_tokens": [tokenizer.decode([tid]) for tid in input_ids],
        "sequence_length": len(input_ids),
        "lngram_config": {
            "use_lngram": True,
            "bits_per_route": bits_per_route,
            "route_count": route_count,
            "alphabet_size": alphabet_size,
            "orders": list(lngram[0].orders),
            "route_memory_width": lngram[0].route_memory_width,
            "table_row_counts": {str(k): int(v) for k, v in table_row_counts.items()},
        },
        "hash_engram_config": {
            "orders": list(hash_engram.orders),
            "heads": hash_engram.heads,
            "table_rows": {str(k): int(v) for k, v in hash_engram.table_rows.items()},
        },
        "namespace_separation": {
            "hash_engram_tables": len(hash_table_params),
            "lngram_tables": len(lngram_table_params),
            "disjoint": namespace_disjoint,
        },
        "forward_verification": {
            "logits_shape": list(logits.shape),
            "hidden_shape": list(hidden.shape),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "hidden_finite": bool(torch.isfinite(hidden).all()),
            "rounds_executed": stats.rounds_executed,
            "block_visits": len(stats.visits),
        },
        "block_lngram_reports": block_lngram_reports,
        "gradient_report": gradient_report,
        "materialization_stats": materialized["stats"],
        "overall_pass": bool(
            all_gradients_ok
            and namespace_disjoint
            and torch.isfinite(logits).all()
            and torch.isfinite(hidden).all()
            and all(
                all(
                    report.get("orders", {}).get(str(order), {}).get("in_bounds", False)
                    for order in (2, 3)
                )
                for report in block_lngram_reports
                if report.get("lngram_active", False)
            )
        ),
    }

    # Cleanup
    surface_index.close()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return result


@app.local_entrypoint()
def main() -> None:
    result = verify_lngram.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
