"""Integration tests for the shared/private working-LoRA decomposition.

Verifies that the gather-then-compute path in ComposedWorkingLoRA.apply_working_lora
produces the same result as explicitly materializing:

    ΔW_{s,b} = Σ_j q_{s,j} b_{b,j} a_{b,j}^T + B_{s,b}^priv A_{s,b}^priv
    Δh = Σ_s g_{t,s} ΔW_{s,b} h

Cross-pair leakage is structurally impossible: the shared basis is common
(not factor-mixed) and private LoRAs are per-skill independent.

Also tests slot population masking and per-skill private rank masking.
"""

from __future__ import annotations

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class ComposedWorkingLoRATests(unittest.TestCase):
    def test_gathered_runtime_matches_explicit_shared_private_equation(self) -> None:
        """The gather-then-compute path must equal the explicit shared+private equation."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        torch.manual_seed(137)
        width = 32
        max_skill_slots = 5
        shared_basis_count = 4
        max_private_lora_rank = 3
        top_k = 2
        block_count = 2
        batch, seq_len = 2, 6

        layer = ComposedWorkingLoRA(
            width,
            max_skill_slots=max_skill_slots,
            shared_basis_count=shared_basis_count,
            max_private_lora_rank=max_private_lora_rank,
            top_k=top_k,
            block_count=block_count,
            init_mode="smoke",
        )
        layer.eval()

        hidden = torch.randn(batch, seq_len, width)

        for block_index in range(block_count):
            # Run the gather-then-compute path
            routing = layer.route(hidden, block_index=block_index)
            delta_factored = layer.apply_working_lora(
                hidden, routing, block_index=block_index
            )

            # Explicit construction:
            # Δh[b,t] = Σ_s g_{t,s} * (Σ_j q_{s,j} b_j (a_j^T h) + B_s^priv (A_s^priv h))
            sa = layer.shared_alpha[block_index]    # [K, D]
            sb = layer.shared_beta[block_index]     # [D, K]
            coeffs = layer.skill_coeffs             # [S, K]
            pa = layer.private_alpha[block_index]   # [S, R_max, D]
            pb = layer.private_beta[block_index]    # [S, D, R_max]
            dense_weights = routing.skill_dense_weights  # [B, T, S]

            delta_explicit = torch.zeros_like(hidden)
            for b in range(batch):
                for t in range(seq_len):
                    h_vec = hidden[b, t]  # [D]
                    for s in range(max_skill_slots):
                        g_s = dense_weights[b, t, s]
                        if g_s == 0:
                            continue
                        # Shared: Σ_j q_{s,j} b_j (a_j^T h)
                        low_shared = sa @ h_vec  # [K]
                        shared_contrib = sb @ (coeffs[s] * low_shared)  # [D]
                        # Private: B_s^priv (A_s^priv h)
                        low_priv = pa[s] @ h_vec  # [R_max]
                        priv_contrib = pb[s] @ low_priv  # [D]
                        delta_explicit[b, t] += g_s * (shared_contrib + priv_contrib)

            self.assertTrue(
                torch.allclose(delta_factored, delta_explicit, atol=1e-4),
                f"Block {block_index}: gathered path does not match explicit equation. "
                f"Max diff: {(delta_factored - delta_explicit).abs().max().item()}",
            )

    def test_production_mode_raises_without_loaded_factors(self) -> None:
        """Production mode must raise RuntimeError until skill factors are loaded."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        layer = ComposedWorkingLoRA(
            16,
            max_skill_slots=3,
            shared_basis_count=2,
            max_private_lora_rank=2,
            top_k=2,
            init_mode="production",
        )
        hidden = torch.randn(1, 4, 16)

        # Must raise before any assets are loaded
        with self.assertRaises(RuntimeError):
            layer.route(hidden, block_index=0)

        # Load skill factors — should now work
        shared_alpha = torch.randn(2, 2, 16)
        shared_beta = torch.randn(2, 16, 2)
        skill_coeffs = torch.randn(3, 2)
        private_alpha = torch.randn(2, 3, 2, 16)
        private_beta = torch.randn(2, 3, 16, 2)
        anchors = torch.randn(3, 16)
        layer.load_skill_factors(
            shared_alpha, shared_beta, skill_coeffs,
            private_alpha, private_beta, anchors,
        )
        routing = layer.route(hidden, block_index=0)
        self.assertEqual(routing.skill_indices.shape, (1, 4, 2))

    def test_production_mode_freezes_anchors(self) -> None:
        """Production mode must freeze anchors after loading."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        layer = ComposedWorkingLoRA(
            16,
            max_skill_slots=3,
            shared_basis_count=2,
            max_private_lora_rank=2,
            top_k=2,
            init_mode="production",
        )
        # Before loading: anchors are trainable
        self.assertTrue(layer.anchors.requires_grad)

        shared_alpha = torch.randn(2, 2, 16)
        shared_beta = torch.randn(2, 16, 2)
        skill_coeffs = torch.randn(3, 2)
        private_alpha = torch.randn(2, 3, 2, 16)
        private_beta = torch.randn(2, 3, 16, 2)
        anchors = torch.randn(3, 16)
        layer.load_skill_factors(
            shared_alpha, shared_beta, skill_coeffs,
            private_alpha, private_beta, anchors,
        )
        # After loading: anchors are frozen
        self.assertFalse(layer.anchors.requires_grad)

    def test_routing_context_is_reused_for_expert_selection(self) -> None:
        """RoutingContext from route() must be usable for expert selection."""
        import torch

        from dendritron.working_adapter import (
            ConditionalExpertBank,
            ComposedWorkingLoRA,
        )

        torch.manual_seed(91)
        width = 16
        max_skill_slots = 4
        shared_basis_count = 3
        max_private_lora_rank = 2
        skill_top_k = 2
        expert_count = 3
        expert_top_k = 2

        skills = ComposedWorkingLoRA(
            width,
            max_skill_slots=max_skill_slots,
            shared_basis_count=shared_basis_count,
            max_private_lora_rank=max_private_lora_rank,
            top_k=skill_top_k,
            init_mode="smoke",
        )
        experts = ConditionalExpertBank(
            width,
            skill_count=max_skill_slots,
            expert_count=expert_count,
            hidden_width=32,
            branch_count=1,
            top_k=expert_top_k,
            init_mode="smoke",
        )

        hidden = torch.randn(1, 5, width)
        routing = skills.route(hidden, block_index=0)

        # Expert selection uses the same routing context
        indices, weights, dense = experts.select_experts(
            hidden, routing=routing
        )
        self.assertEqual(indices.shape, (1, 5, expert_top_k))
        self.assertTrue(torch.isfinite(weights).all())

    def test_smoke_mode_label_is_explicit(self) -> None:
        """Smoke mode must be explicitly labeled in init_mode."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        layer = ComposedWorkingLoRA(
            8,
            max_skill_slots=2,
            shared_basis_count=2,
            max_private_lora_rank=1,
            top_k=1,
            init_mode="smoke",
        )
        self.assertEqual(layer.init_mode, "smoke")
        self.assertTrue(bool(layer.factors_loaded))

    def test_skill_mask_zeros_inactive_skills(self) -> None:
        """Inactive skills must contribute zero to the working LoRA output."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        torch.manual_seed(203)
        width = 8
        max_skill_slots = 3
        shared_basis_count = 2
        max_private_lora_rank = 2
        top_k = 1  # Only 1 skill active per position

        layer = ComposedWorkingLoRA(
            width,
            max_skill_slots=max_skill_slots,
            shared_basis_count=shared_basis_count,
            max_private_lora_rank=max_private_lora_rank,
            top_k=top_k,
            init_mode="smoke",
        )
        layer.eval()

        hidden = torch.randn(1, 4, width)
        routing = layer.route(hidden, block_index=0)

        # With top_k=1, only 1 skill is active per position
        active_per_position = routing.skill_mask.sum(dim=-1)
        self.assertTrue(bool((active_per_position == 1).all()))

        # Verify that masking out ALL skills produces zero output
        zero_routing = type(routing)(
            skill_indices=routing.skill_indices,
            skill_weights=torch.zeros_like(routing.skill_weights),
            skill_dense_weights=torch.zeros_like(routing.skill_dense_weights),
            skill_mask=torch.zeros_like(routing.skill_mask),
        )
        delta_zero = layer.apply_working_lora(
            hidden, zero_routing, block_index=0
        )
        self.assertTrue(torch.allclose(delta_zero, torch.zeros_like(hidden), atol=1e-7))

    def test_zero_cross_skill_leakage(self) -> None:
        """With only skill s active, the output must equal ΔW_s h exactly.

        No other skill's private LoRA or coefficients may contribute.
        The shared basis is common to all skills, but only the active
        skill's coefficients q_s and private LoRA (B_s^priv, A_s^priv)
        participate.
        """
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA, RoutingContext

        torch.manual_seed(314)
        width = 16
        max_skill_slots = 4
        shared_basis_count = 3
        max_private_lora_rank = 2
        top_k = 1

        layer = ComposedWorkingLoRA(
            width,
            max_skill_slots=max_skill_slots,
            shared_basis_count=shared_basis_count,
            max_private_lora_rank=max_private_lora_rank,
            top_k=top_k,
            init_mode="smoke",
        )
        layer.eval()

        hidden = torch.randn(2, 5, width)

        for block_index in range(2):
            sa = layer.shared_alpha[block_index]    # [K, D]
            sb = layer.shared_beta[block_index]     # [D, K]
            coeffs = layer.skill_coeffs             # [S, K]
            pa = layer.private_alpha[block_index]   # [S, R_max, D]
            pb = layer.private_beta[block_index]    # [S, D, R_max]

            for target_skill in range(max_skill_slots):
                # Manually construct a routing context that activates only target_skill
                indices = torch.full(
                    (2, 5, top_k), target_skill, dtype=torch.long
                )
                weights = torch.ones(2, 5, top_k)
                dense = torch.zeros(2, 5, max_skill_slots)
                dense[:, :, target_skill] = 1.0
                mask = dense != 0

                routing = RoutingContext(
                    skill_indices=indices,
                    skill_weights=weights,
                    skill_dense_weights=dense,
                    skill_mask=mask,
                )

                delta = layer.apply_working_lora(
                    hidden, routing, block_index=block_index
                )

                # Explicit: ΔW_s h = (Σ_j q_{s,j} b_j a_j^T + B_s^priv A_s^priv) h
                for b in range(2):
                    for t in range(5):
                        h_vec = hidden[b, t]
                        # Shared: sb @ (q_s ⊙ (sa @ h))
                        low_shared = sa @ h_vec  # [K]
                        shared_contrib = sb @ (coeffs[target_skill] * low_shared)  # [D]
                        # Private: pb_s @ (pa_s @ h)
                        low_priv = pa[target_skill] @ h_vec  # [R_max]
                        priv_contrib = pb[target_skill] @ low_priv  # [D]
                        expected = shared_contrib + priv_contrib
                        self.assertTrue(
                            torch.allclose(delta[b, t], expected, atol=1e-4),
                            f"Block {block_index}, skill {target_skill}, "
                            f"token ({b},{t}): cross-skill leakage detected. "
                            f"Max diff: {(delta[b, t] - expected).abs().max().item()}",
                        )

    def test_skill_coeffs_shared_across_blocks(self) -> None:
        """skill_coeffs must be [S, K], shared across blocks (not block-specific)."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        layer = ComposedWorkingLoRA(
            16,
            max_skill_slots=4,
            shared_basis_count=3,
            max_private_lora_rank=2,
            top_k=2,
            block_count=2,
            init_mode="smoke",
        )
        # skill_coeffs shape: [S, K] — not [block_count, S, K]
        self.assertEqual(layer.skill_coeffs.shape, (4, 3))

    def test_slot_population_mask_excludes_unpopulated_slots(self) -> None:
        """Unpopulated skill slots must not be selected by routing."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        torch.manual_seed(555)
        width = 16
        max_skill_slots = 6
        shared_basis_count = 3
        max_private_lora_rank = 2
        top_k = 3

        layer = ComposedWorkingLoRA(
            width,
            max_skill_slots=max_skill_slots,
            shared_basis_count=shared_basis_count,
            max_private_lora_rank=max_private_lora_rank,
            top_k=top_k,
            init_mode="smoke",
        )
        layer.eval()

        # Mark slots 3, 4, 5 as unpopulated
        slot_populated = torch.tensor([True, True, True, False, False, False])
        layer.slot_populated.copy_(slot_populated)

        hidden = torch.randn(2, 8, width)
        routing = layer.route(hidden, block_index=0)

        # No unpopulated slot should appear in the selected indices
        selected = routing.skill_indices  # [B, T, top_k]
        for b in range(2):
            for t in range(8):
                for k in range(top_k):
                    self.assertTrue(
                        slot_populated[selected[b, t, k]],
                        f"Unpopulated slot {selected[b, t, k].item()} selected at ({b},{t},{k})"
                    )

    def test_private_rank_mask_zeros_padded_dimensions(self) -> None:
        """Per-skill private rank mask must zero out padded private LoRA dimensions."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA, RoutingContext

        torch.manual_seed(777)
        width = 16
        max_skill_slots = 3
        shared_basis_count = 2
        max_private_lora_rank = 4
        top_k = 1

        layer = ComposedWorkingLoRA(
            width,
            max_skill_slots=max_skill_slots,
            shared_basis_count=shared_basis_count,
            max_private_lora_rank=max_private_lora_rank,
            top_k=top_k,
            init_mode="smoke",
        )
        layer.eval()

        # Set slot 0 to rank 2 (padded dims 2,3 should be masked)
        # Set slot 1 to rank 1 (padded dims 1,2,3 should be masked)
        # Set slot 2 to full rank 4
        layer.private_ranks.copy_(torch.tensor([2, 1, 4]))

        hidden = torch.randn(1, 4, width)

        for block_index in range(2):
            pa = layer.private_alpha[block_index]   # [S, R_max, D]
            pb = layer.private_beta[block_index]    # [S, D, R_max]

            for target_slot in range(max_skill_slots):
                actual_rank = int(layer.private_ranks[target_slot].item())

                # Manually activate only target_slot
                indices = torch.full((1, 4, top_k), target_slot, dtype=torch.long)
                weights = torch.ones(1, 4, top_k)
                dense = torch.zeros(1, 4, max_skill_slots)
                dense[:, :, target_slot] = 1.0
                mask = dense != 0

                routing = RoutingContext(
                    skill_indices=indices,
                    skill_weights=weights,
                    skill_dense_weights=dense,
                    skill_mask=mask,
                )

                delta = layer.apply_working_lora(
                    hidden, routing, block_index=block_index
                )

                # Explicit with rank masking: only first actual_rank dims contribute
                sa = layer.shared_alpha[block_index]
                sb = layer.shared_beta[block_index]
                coeffs = layer.skill_coeffs

                for t in range(4):
                    h_vec = hidden[0, t]
                    # Shared (unaffected by rank mask)
                    low_shared = sa @ h_vec
                    shared_contrib = sb @ (coeffs[target_slot] * low_shared)
                    # Private with rank mask
                    low_priv_full = pa[target_slot] @ h_vec  # [R_max]
                    # Zero out padded dims
                    low_priv_masked = low_priv_full.clone()
                    low_priv_masked[actual_rank:] = 0
                    priv_contrib = pb[target_slot] @ low_priv_masked
                    expected = shared_contrib + priv_contrib

                    self.assertTrue(
                        torch.allclose(delta[0, t], expected, atol=1e-4),
                        f"Block {block_index}, slot {target_slot} (rank {actual_rank}), "
                        f"token {t}: rank mask not applied correctly. "
                        f"Max diff: {(delta[0, t] - expected).abs().max().item()}",
                    )

    def test_production_mode_starts_with_no_slots_populated(self) -> None:
        """Production mode must start with all slots unpopulated."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        layer = ComposedWorkingLoRA(
            16,
            max_skill_slots=4,
            shared_basis_count=2,
            max_private_lora_rank=2,
            top_k=2,
            init_mode="production",
        )
        self.assertFalse(bool(layer.slot_populated.any()))
        self.assertTrue(bool((layer.private_ranks == 0).all()))

    def test_smoke_mode_all_slots_populated(self) -> None:
        """Smoke mode must start with all slots populated at max rank."""
        import torch

        from dendritron.working_adapter import ComposedWorkingLoRA

        layer = ComposedWorkingLoRA(
            16,
            max_skill_slots=4,
            shared_basis_count=2,
            max_private_lora_rank=3,
            top_k=2,
            init_mode="smoke",
        )
        self.assertTrue(bool(layer.slot_populated.all()))
        self.assertTrue(bool((layer.private_ranks == 3).all()))


if __name__ == "__main__":
    unittest.main()
