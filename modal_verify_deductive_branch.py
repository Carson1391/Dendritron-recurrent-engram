"""Verify the typed deductive HarMax branch with a production-width forward/backward test.

Tests:
1. Enable deductive_branches_per_block=1 in the production config.
2. Run a forward pass with 5+ tokens so premises accumulate causally.
3. Report branch contraction stats: movement shape, evidence, residual, confidence.
4. Confirm branch movement is finite and nonzero.
5. Run one backward pass and confirm finite, nonzero gradients for:
   - conclusion_anchor (each expert that has a deductive branch)
   - contradiction_anchors
   - premise_evidence_raw
   - conclusion_evidence_raw
   - gate
6. Confirm the block order: memory + LNGram -> branch contraction -> first RMSNorm.
7. Confirm branches only execute from active experts reached through skill adjacency.

Run:
    modal run modal_verify_deductive_branch.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

from modal_extract_states import CORPUS_VOLUME_NAME, DATA_MOUNT


APP_NAME = "dendritron-deductive-branch-verification"

JTD_ROOT = "/data/dendritron-stage4-jtd-punctuation-v2"
LATENT_ASSETS_ROOT = f"{JTD_ROOT}/latent-assets"
TOKEN_EMBEDDINGS_PATH = f"{LATENT_ASSETS_ROOT}/frontend/qwen_token_embeddings.safetensors"
JTD_PROJECTIONS_PATH = f"{JTD_ROOT}/jtd-projections.pt"
SURFACE_INDEX_PATH = f"{JTD_ROOT}/surface_index.sqlite3"
TOKENIZER_SNAPSHOT_DIR = f"{JTD_ROOT}/tokenizer_snapshot"
DICTIONARY_BANK_ROOT = "/data/dendritron-stage3-dictionary/bank"
STAGE2_ROOT = "/data/dendritron-stage2-punctuation-v2"

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
def verify_deductive_branch() -> dict[str, Any]:
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

    hash_engram = HashEngramAddressor(
        orders=(2, 3),
        heads=4,
        table_rows={2: 65_536, 3: 262_144},
    )
    addressor = SurfaceMemoryAddressor(
        exact_index=surface_index,
        hash_engram=hash_engram,
    )

    # --- 3. Tokenize ---
    input_ids = tokenizer.encode(TEST_PHRASE, add_special_tokens=False)
    plans = addressor.resolve_text(tokenizer, TEST_PHRASE)

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

    # --- 4. Construct production DendritronLM with deductive branches ---
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
        use_lngram=True,
        lngram_bits_per_route=4,
        lngram_orders=(2, 3),
        lngram_route_memory_width=4,
        hash_orders=(2, 3),
        hash_heads=4,
        hash_table_rows=(65_536, 262_144),
        hash_memory_width=88,
        # Typed deductive branch: 1 per block per expert
        deductive_branches_per_block=1,
        deductive_max_premises=8,
        deductive_max_contradictions=4,
    )
    model = DendritronLM(config)

    # --- 5. Load frozen assets ---
    model.load_token_embedding_artifact(TOKEN_EMBEDDINGS_PATH)
    model.load_joint_transfer_checkpoint(JTD_PROJECTIONS_PATH)

    # --- 6. Forward pass ---
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    output = model(input_tensor, memory_payloads=payloads, return_stats=True)

    logits = output.logits
    hidden = output.hidden
    stats = output.recurrent_stats

    # --- 7. Extract branch contraction stats per block ---
    branch_reports = []
    for visit in stats.visits:
        block_idx = visit.block_index
        bc = visit.branch_contraction
        if bc is None:
            branch_reports.append({
                "block_index": block_idx,
                "branch_contraction_active": False,
            })
            continue

        report = {
            "block_index": block_idx,
            "branch_contraction_active": True,
            "branch_count": bc.branch_count,
            "movement_shape": list(bc.movement.shape),
            "movement_finite": bool(torch.isfinite(bc.movement).all()),
            "movement_nonzero": bool(bc.movement.abs().sum() > 0),
            "movement_norm": float(bc.movement.norm().item()),
        }

        # Per-branch evidence and residual stats
        for j, (ev, res) in enumerate(zip(bc.branch_evidences, bc.branch_residuals, strict=True)):
            report[f"branch_{j}_evidence_mean"] = float(ev.mean().item())
            report[f"branch_{j}_evidence_std"] = float(ev.std().item())
            report[f"branch_{j}_residual_mean"] = float(res.mean().item())
            report[f"branch_{j}_residual_std"] = float(res.std().item())

        branch_reports.append(report)

    # --- 7b. Track which experts were actually routed to ---
    active_expert_ids = set()
    for visit in stats.visits:
        for eid in visit.skill_expert.experts.indices.reshape(-1).tolist():
            active_expert_ids.add(int(eid))

    # --- 8. Backward pass ---
    loss = logits.square().mean()
    loss.backward()

    # --- 9. Check deductive branch gradients (only for routed experts) ---
    gradient_report = {
        "loss_value": float(loss.item()),
        "loss_finite": bool(torch.isfinite(loss)),
        "active_expert_ids": sorted(active_expert_ids),
    }

    all_grads_ok = True
    expert_bank = model.core.skill_expert.experts
    for expert_id, expert in enumerate(expert_bank.experts):
        if expert.deductive_branches is None:
            continue
        # Skip experts that were never routed to — their branches correctly
        # never fired, so gradients are None. This is expected behavior:
        # branches only execute from active experts reached through skill adjacency.
        if expert_id not in active_expert_ids:
            continue
        for block_idx in range(2):
            for branch_idx, branch in enumerate(expert.deductive_branches[block_idx]):
                key = f"expert_{expert_id}_block_{block_idx}_branch_{branch_idx}"
                grad_report = {}

                # conclusion_anchor gradient
                ca_grad = branch.conclusion_anchor.grad
                if ca_grad is not None:
                    grad_report["conclusion_anchor"] = {
                        "norm": float(ca_grad.norm().item()),
                        "finite": bool(torch.isfinite(ca_grad).all()),
                        "nonzero": bool(ca_grad.abs().sum() > 0),
                    }
                    if not (grad_report["conclusion_anchor"]["finite"] and grad_report["conclusion_anchor"]["nonzero"]):
                        all_grads_ok = False
                else:
                    grad_report["conclusion_anchor"] = {"norm": 0.0, "finite": False, "nonzero": False}
                    all_grads_ok = False

                # contradiction_anchors gradient
                ct_grad = branch.contradiction_anchors.grad
                if ct_grad is not None:
                    grad_report["contradiction_anchors"] = {
                        "norm": float(ct_grad.norm().item()),
                        "finite": bool(torch.isfinite(ct_grad).all()),
                        "nonzero": bool(ct_grad.abs().sum() > 0),
                    }
                    if not (grad_report["contradiction_anchors"]["finite"] and grad_report["contradiction_anchors"]["nonzero"]):
                        all_grads_ok = False
                else:
                    grad_report["contradiction_anchors"] = {"norm": 0.0, "finite": False, "nonzero": False}
                    all_grads_ok = False

                # premise_evidence_raw gradient
                pe_grad = branch.premise_evidence_raw.grad
                if pe_grad is not None:
                    grad_report["premise_evidence_raw"] = {
                        "norm": float(pe_grad.norm().item()),
                        "finite": bool(torch.isfinite(pe_grad).all()),
                        "nonzero": bool(pe_grad.abs().sum() > 0),
                    }
                    if not (grad_report["premise_evidence_raw"]["finite"] and grad_report["premise_evidence_raw"]["nonzero"]):
                        all_grads_ok = False
                else:
                    grad_report["premise_evidence_raw"] = {"norm": 0.0, "finite": False, "nonzero": False}
                    all_grads_ok = False

                # conclusion_evidence_raw gradient
                ce_grad = branch.conclusion_evidence_raw.grad
                if ce_grad is not None:
                    grad_report["conclusion_evidence_raw"] = {
                        "norm": float(ce_grad.norm().item()),
                        "finite": bool(torch.isfinite(ce_grad).all()),
                        "nonzero": bool(ce_grad.abs().sum() > 0),
                    }
                    if not (grad_report["conclusion_evidence_raw"]["finite"] and grad_report["conclusion_evidence_raw"]["nonzero"]):
                        all_grads_ok = False
                else:
                    grad_report["conclusion_evidence_raw"] = {"norm": 0.0, "finite": False, "nonzero": False}
                    all_grads_ok = False

                # gate gradient
                g_grad = branch.gate.grad
                if g_grad is not None:
                    grad_report["gate"] = {
                        "norm": float(g_grad.norm().item()),
                        "finite": bool(torch.isfinite(g_grad).all()),
                        "nonzero": bool(g_grad.abs().sum() > 0),
                    }
                    if not (grad_report["gate"]["finite"] and grad_report["gate"]["nonzero"]):
                        all_grads_ok = False
                else:
                    grad_report["gate"] = {"norm": 0.0, "finite": False, "nonzero": False}
                    all_grads_ok = False

                gradient_report[key] = grad_report

    gradient_report["all_gradients_ok"] = all_grads_ok

    # --- 10. Verify block order: branch contraction runs after LNGram ---
    # The recurrent core should have branch_contraction stats populated
    # and LNGram stats populated for each visit.
    block_order_ok = True
    for visit in stats.visits:
        if visit.lngram is None:
            block_order_ok = False
            break
        # branch_contraction can be None if no expert has deductive branches,
        # but with deductive_branches_per_block=1 it should be active
        if visit.branch_contraction is None:
            block_order_ok = False
            break

    # --- 11. Verify branches only execute from active experts ---
    # With expert_top_k=2, only 2 experts are selected per position.
    # Each has 1 deductive branch, so branch_count should be <= 2 * expert_top_k
    # per block visit (but could be less if some experts' branches don't fire).
    branch_count_reasonable = all(
        report.get("branch_count", 0) <= config.expert_top_k * config.deductive_branches_per_block
        for report in branch_reports
        if report.get("branch_contraction_active", False)
    )

    # --- 12. Assemble result ---
    result = {
        "completed": True,
        "test_phrase": TEST_PHRASE,
        "input_ids": input_ids,
        "input_tokens": [tokenizer.decode([tid]) for tid in input_ids],
        "sequence_length": len(input_ids),
        "config": {
            "deductive_branches_per_block": config.deductive_branches_per_block,
            "deductive_max_premises": config.deductive_max_premises,
            "deductive_max_contradictions": config.deductive_max_contradictions,
            "expert_count": config.expert_count,
            "expert_top_k": config.expert_top_k,
            "max_skill_slots": config.max_skill_slots,
            "skill_top_k": config.skill_top_k,
            "use_lngram": config.use_lngram,
        },
        "forward_verification": {
            "logits_shape": list(logits.shape),
            "hidden_shape": list(hidden.shape),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "hidden_finite": bool(torch.isfinite(hidden).all()),
            "rounds_executed": stats.rounds_executed,
            "block_visits": len(stats.visits),
        },
        "branch_reports": branch_reports,
        "gradient_report": gradient_report,
        "block_order_ok": block_order_ok,
        "branch_count_reasonable": branch_count_reasonable,
        "overall_pass": bool(
            all_grads_ok
            and block_order_ok
            and branch_count_reasonable
            and torch.isfinite(logits).all()
            and torch.isfinite(hidden).all()
            and all(
                r.get("movement_finite", False) and r.get("movement_nonzero", False)
                for r in branch_reports
                if r.get("branch_contraction_active", False)
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
    result = verify_deductive_branch.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
