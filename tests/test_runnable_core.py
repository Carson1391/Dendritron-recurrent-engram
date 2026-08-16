from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class RunnableCoreTests(unittest.TestCase):
    def test_two_blocks_execute_two_sublayers_and_decode(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        vocab_size = 128
        model = DendritronLM(tiny_smoke_config(vocab_size))
        token_ids = torch.tensor([[17, 23, 41, 9, 52]], dtype=torch.long)
        output = model(token_ids, return_stats=True)
        self.assertEqual(
            tuple(output.logits.shape),
            (1, token_ids.shape[1], vocab_size),
        )
        self.assertEqual(output.recurrent_stats.rounds_executed, 2)
        self.assertEqual(len(output.recurrent_stats.visits), 4)
        self.assertEqual(
            [visit.block_index for visit in output.recurrent_stats.visits],
            [0, 1, 0, 1],
        )
        self.assertEqual(len(model.core.contraction_norms), 2)
        self.assertEqual(len(model.core.compute_norms), 2)

    def test_harmax_contraction_is_causal(self) -> None:
        import torch

        from dendritron.geometric_attention import HarMaxContraction

        torch.manual_seed(7)
        layer = HarMaxContraction(
            16,
            max_sequence_length=8,
            candidate_window=8,
            top_k=4,
        )
        prefix = torch.randn(1, 4, 16)
        changed = prefix.clone()
        changed[:, 3] = torch.randn(16)
        left = layer(prefix)
        right = layer(changed)
        self.assertTrue(torch.allclose(left[:, :3], right[:, :3], atol=1e-6))

    def test_harmax_pool_reports_attraction_and_repulsion(self) -> None:
        import torch

        from dendritron.geometric_attention import HarMaxContraction

        layer = HarMaxContraction(
            4,
            max_sequence_length=4,
            candidate_window=2,
            top_k=2,
        )
        query = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        anchors = torch.tensor(
            [[[[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]]]]
        )
        _, stats = layer.contract_pool(
            query,
            anchors,
            evidence=torch.tensor([[[1.0, 1.0]]]),
            supported=torch.tensor([[[True, False]]]),
        )
        self.assertGreater(float(stats.signed_coefficients[0, 0, 0]), 0.0)
        self.assertLess(float(stats.signed_coefficients[0, 0, 1]), 0.0)
        self.assertGreater(float(stats.attraction_mass), 0.0)
        self.assertGreater(float(stats.repulsion_mass), 0.0)
        self.assertTrue(torch.isfinite(stats.harmonic_residual).all())

    def test_sparse_memory_changes_live_scores_when_enabled(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.memory_fusion import MemoryPayloads
        from dendritron.model import DendritronLM

        torch.manual_seed(11)
        model = DendritronLM(tiny_smoke_config(vocab_size=128))
        token_ids = torch.tensor([[11, 22, 33, 44, 55, 66]], dtype=torch.long)
        batch, length = token_ids.shape
        width = model.config.memory_width
        payloads = MemoryPayloads(
            phrase_layer8=torch.randn(batch, length, width),
            phrase_layer24=torch.randn(batch, length, width),
            phrase_mask=torch.ones(batch, length, dtype=torch.bool),
            definitions=torch.randn(batch, length, 2, width),
            definition_mask=torch.ones(batch, length, 2, dtype=torch.bool),
            definition_sense_rows=torch.arange(
                batch * length * 2,
                dtype=torch.long,
            ).view(batch, length, 2),
            hash_addresses={
                2: torch.randint(
                    0,
                    model.config.hash_rows_by_order[2],
                    (batch, length, model.config.hash_heads),
                ),
                3: torch.randint(
                    0,
                    model.config.hash_rows_by_order[3],
                    (batch, length, model.config.hash_heads),
                ),
            },
        )
        baseline = model(token_ids).logits
        with torch.no_grad():
            model.memory_fusion.initial_phrase8_gate.fill_(0.7)
            model.memory_fusion.initial_hash_gate.fill_(0.7)
            model.memory_fusion.phrase8_gates.fill_(0.7)
            model.memory_fusion.phrase24_gates.fill_(0.7)
            model.memory_fusion.definition_gates.fill_(0.7)
            model.memory_fusion.hash_gates.fill_(0.7)
            for gate in model.memory_fusion.hash_memory.order_gates.values():
                gate.fill_(0.7)
        with_memory = model(token_ids, memory_payloads=payloads).logits
        self.assertFalse(torch.allclose(baseline, with_memory))

    def test_definitions_keep_exact_joint_locations(self) -> None:
        import torch

        from dendritron.joint_transfer import JointTransferDomain

        jtd = JointTransferDomain(4, memory_width=4)
        definitions = torch.randn(2, 3, 5, 4)
        joint = jtd.definitions_to_joint(definitions)
        self.assertEqual(joint.data_ptr(), definitions.data_ptr())
        self.assertTrue(torch.equal(joint, definitions))

    def test_all_lookup_senses_remain_in_continuous_concept_field(self) -> None:
        import torch

        from dendritron.memory_fusion import SparseMemoryFusion

        fusion = SparseMemoryFusion(
            4,
            memory_width=4,
            hash_rows_by_order={2: 8, 3: 8},
            hash_heads=1,
            hash_memory_width=2,
        )
        hidden = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        definitions = torch.tensor(
            [[[[1.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0],
               [-1.0, 0.0, 0.0, 0.0]]]]
        )
        mask = torch.ones(1, 1, 3, dtype=torch.bool)
        rows = torch.tensor([[[101, 102, 103]]], dtype=torch.long)
        update, weights, distances, returned_rows, signed_coeff = fusion._definition_field(
            hidden,
            definitions,
            mask,
            rows,
        )
        self.assertEqual(tuple(update.shape), tuple(hidden.shape))
        self.assertTrue((weights > 0).all())
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1, 1)))
        self.assertLess(float(distances[0, 0, 0]), float(distances[0, 0, 2]))
        self.assertTrue(torch.equal(returned_rows, rows))

    def test_definition_geometry_enters_after_causal_context(self) -> None:
        import torch

        from dendritron.memory_fusion import MemoryPayloads, SparseMemoryFusion

        fusion = SparseMemoryFusion(
            4,
            memory_width=4,
            hash_rows_by_order={2: 8, 3: 8},
            hash_heads=1,
            hash_memory_width=2,
        )
        payloads = MemoryPayloads(
            definitions=torch.randn(1, 2, 2, 4),
            definition_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        )
        self.assertTrue(
            torch.equal(
                fusion.initial_update(torch.randn(1, 2, 4), payloads),
                torch.zeros(1, 2, 4),
            )
        )

    def test_locality_alignment_loss_reaches_reference_geometry(self) -> None:
        import torch

        from dendritron.joint_transfer import locality_preserving_alignment_loss

        reference = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        total, stats = locality_preserving_alignment_loss(
            reference.clone(),
            reference,
            neighbor_count=2,
        )
        self.assertLess(float(total), 1e-8)
        self.assertLess(float(stats.point_loss), 1e-8)
        self.assertLess(float(stats.locality_loss), 1e-8)

    def test_rank_objective_backpropagates_through_both_harmax_blocks(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM
        from dendritron.output_geometry import rank_margin_loss

        model = DendritronLM(tiny_smoke_config(vocab_size=128))
        values = [7, 18, 29, 40, 51, 62, 73, 84, 95]
        inputs = torch.tensor([values[:-1]], dtype=torch.long)
        targets = torch.tensor([values[1:]], dtype=torch.long)
        loss = rank_margin_loss(
            model(inputs).logits,
            targets,
            hard_negatives=8,
        )
        loss.backward()
        for block in model.core.harmax:
            self.assertIsNotNone(block.relative_evidence.grad)
            self.assertGreater(float(block.relative_evidence.grad.abs().sum()), 0.0)
            self.assertIsNotNone(block.relative_penalty_raw.grad)

    def test_skill_expert_and_hash_paths_receive_gradients(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.memory_fusion import MemoryPayloads
        from dendritron.model import DendritronLM

        torch.manual_seed(29)
        model = DendritronLM(tiny_smoke_config(vocab_size=128))
        token_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        payloads = MemoryPayloads(
            hash_addresses={
                order: torch.randint(
                    0,
                    rows,
                    (1, 5, model.config.hash_heads),
                )
                for order, rows in model.config.hash_rows_by_order.items()
            }
        )
        model(token_ids, memory_payloads=payloads).logits.square().mean().backward()
        # In the paired working LoRA, alpha_all/beta_all are frozen;
        # the trainable parameters are anchors, skill_gate, and expert_gate.
        self.assertGreater(
            float(model.core.skill_expert.skills.anchors.grad.abs().sum()),
            0.0,
        )
        expert_gradient = sum(
            float(parameter.grad.abs().sum())
            for name, parameter in model.core.skill_expert.experts.named_parameters()
            if name.endswith("output.weight") and parameter.grad is not None
        )
        self.assertGreater(expert_gradient, 0.0)
        hash_gradient = sum(
            float(table.weight.grad.abs().sum())
            for table in model.memory_fusion.hash_memory.tables.values()
            if table.weight.grad is not None
        )
        self.assertGreater(hash_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
