"""Tests for DefinitionLnGram, signed HarMax _definition_field, and pipeline wiring.

Verifies:
1. DefinitionLnGram constructs with correct shapes and config
2. attach_definition_bank validates shape and rejects mismatched widths
3. Forward without bank raises RuntimeError
4. Forward with bank produces output of correct shape
5. Output is a residual update (output != input, same shape)
6. Signed HarMax movement: attraction toward high-evidence anchors
7. Signed HarMax movement: repulsion from low-evidence anchors
8. Counterfactual W_q backward: gradient flows to address_projection weights
9. Counterfactual W_q backward: gradient is non-zero
10. AddressRecordTable populate + lookup integration
11. DefinitionLnGram stats contain symbols, addresses, movement
12. _definition_field signed HarMax: evidence drives attraction vs repulsion
13. _definition_field returns 5-tuple with signed_coeff
14. _definition_field with evidence=None defaults to uniform
15. MemoryPayloads carries definition_evidence through .to()
16. End-to-end model forward with bank attached uses DefinitionLnGram path
17. DefinitionLnGram bank is not a trainable parameter
18. J_h^T (training gradient) distinct from joint_to_live (runtime map)
"""

from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class DefinitionLnGramConstructionTests(unittest.TestCase):
    def test_constructs_with_correct_shapes(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        model_width = 2048
        memory_width = 2048
        mod = DefinitionLnGram(
            model_width, memory_width,
            bits_per_route=4, orders=(2, 3), senses_per_address=4,
        )
        self.assertEqual(mod.model_width, model_width)
        self.assertEqual(mod.memory_width, memory_width)
        self.assertEqual(mod.bits_per_route, 4)
        self.assertEqual(mod.route_count, model_width // 4)
        self.assertEqual(mod.alphabet_size, 16)
        self.assertEqual(mod.orders, (2, 3))
        self.assertEqual(mod.senses_per_address, 4)
        self.assertIsNotNone(mod.address_table)
        self.assertIsNotNone(mod.joint_transfer)
        self.assertIsNotNone(mod.input_norm)
        self.assertIsNotNone(mod.address_projection)

    def test_attach_definition_bank_validates_shape(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        # Wrong ndim
        with self.assertRaises(ValueError):
            mod.attach_definition_bank(torch.randn(10, 2048, 1))
        # Wrong width
        with self.assertRaises(ValueError):
            mod.attach_definition_bank(torch.randn(100, 1024))

    def test_forward_without_bank_raises(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        hidden = torch.randn(2, 8, 2048)
        with self.assertRaises(RuntimeError):
            mod(hidden)

    def test_forward_with_bank_produces_correct_shape(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)
        hidden = torch.randn(2, 8, 2048)
        output = mod(hidden)
        self.assertEqual(tuple(output.shape), (2, 8, 2048))

    def test_output_is_residual_update(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)

        # Populate address table entries so lookup finds non-empty senses
        for order in (2, 3):
            rows = mod.address_table.route_count * mod.address_table.alphabet_size**order
            addresses = torch.arange(min(20, rows))
            n = addresses.shape[0]
            mod.address_table.populate(
                order,
                addresses,
                sense_rows=torch.randint(0, 500, (n, 4)),
                mask=torch.ones(n, 4, dtype=torch.bool),
                evidence=torch.ones(n, 4, dtype=torch.float32),
            )

        hidden = torch.randn(1, 6, 2048)
        output = mod(hidden)
        # Output should differ from input (residual update applied)
        self.assertFalse(torch.allclose(output, hidden, atol=1e-8))
        # But same shape
        self.assertEqual(output.shape, hidden.shape)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class SignedHarMaxMovementTests(unittest.TestCase):
    def test_attraction_toward_high_evidence_anchors(self) -> None:
        """When all anchors have equal evidence, movement should pull query
        toward the centroid (attraction)."""
        import torch

        from dendritron.definition_lngram import _signed_harmax_movement

        # Single query, two anchors equidistant from query, equal evidence
        query = torch.zeros(1, 1, 8)
        anchors = torch.tensor([[[[1.0, 0, 0, 0, 0, 0, 0, 0],
                                   [-1.0, 0, 0, 0, 0, 0, 0, 0]]]])
        evidence = torch.tensor([[[1.0, 1.0]]])
        valid = torch.tensor([[[True, True]]])

        movement = _signed_harmax_movement(
            query, [anchors], [evidence], [valid],
            harmonic_exponent=2.0, epsilon=1e-6,
        )
        # Symmetric anchors with equal evidence: net movement ~ 0
        self.assertTrue(movement.abs().max() < 1e-3)

    def test_repulsion_from_low_evidence_anchors(self) -> None:
        """When one anchor has much lower evidence than the other, the signed
        coefficient (y-p) becomes negative for that anchor, producing repulsion."""
        import torch

        from dendritron.definition_lngram import _signed_harmax_movement

        query = torch.zeros(1, 1, 8)
        # Two anchors: one at +x, one at -x
        anchors = torch.tensor([[[[1.0, 0, 0, 0, 0, 0, 0, 0],
                                   [-1.0, 0, 0, 0, 0, 0, 0, 0]]]])
        # Asymmetric evidence: anchor 0 has 10x evidence of anchor 1
        evidence = torch.tensor([[[10.0, 0.01]]])
        valid = torch.tensor([[[True, True]]])

        movement = _signed_harmax_movement(
            query, [anchors], [evidence], [valid],
            harmonic_exponent=2.0, epsilon=1e-6,
        )
        # High evidence on +x anchor should pull query toward +x
        self.assertGreater(movement[0, 0, 0].item(), 0.0)

    def test_signed_coeff_sum_is_zero(self) -> None:
        """y - p sums to zero across valid anchors when evidence is uniform,
        because y and p are both probability distributions."""
        import torch

        from dendritron.definition_lngram import _signed_harmax_movement

        query = torch.randn(1, 1, 4)
        anchors = torch.randn(1, 1, 3, 4)
        evidence = torch.ones(1, 1, 3)
        valid = torch.tensor([[[True, True, True]]])

        # We can't directly access signed_coeff from _signed_harmax_movement,
        # but we can verify that uniform evidence with equidistant anchors
        # produces near-zero movement (symmetric case)
        equidistant = torch.tensor([[[[1.0, 0, 0, 0],
                                       [0, 1.0, 0, 0],
                                       [0, 0, 1.0, 0]]]])
        query_zero = torch.zeros(1, 1, 4)
        movement = _signed_harmax_movement(
            query_zero, [equidistant], [evidence], [valid],
            harmonic_exponent=2.0, epsilon=1e-6,
        )
        # Equal evidence, equal distance -> signed_coeff sums to 0 -> no net movement
        self.assertTrue(movement.abs().max() < 1e-3)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class CounterfactualBackwardTests(unittest.TestCase):
    def test_gradient_flows_to_address_projection(self) -> None:
        """Counterfactual W_q backward should produce non-zero gradient on
        address_projection.weight."""
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)

        # Populate some address table entries so lookup finds non-empty senses
        for order in (2, 3):
            rows = mod.address_table.route_count * mod.address_table.alphabet_size**order
            addresses = torch.arange(min(20, rows))
            n = addresses.shape[0]
            mod.address_table.populate(
                order,
                addresses,
                sense_rows=torch.randint(0, 500, (n, 4)),
                mask=torch.ones(n, 4, dtype=torch.bool),
                evidence=torch.ones(n, 4, dtype=torch.float32),
            )

        hidden = torch.randn(1, 8, 2048, requires_grad=True)
        output = mod(hidden)
        loss = output.sum()
        loss.backward()

        self.assertIsNotNone(mod.address_projection.weight.grad)
        self.assertGreater(mod.address_projection.weight.grad.abs().max().item(), 0.0)

    def test_gradient_is_non_zero(self) -> None:
        """Verify that the counterfactual gradient is not just zeros."""
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)

        for order in (2, 3):
            rows = mod.address_table.route_count * mod.address_table.alphabet_size**order
            addresses = torch.arange(min(20, rows))
            n = addresses.shape[0]
            mod.address_table.populate(
                order,
                addresses,
                sense_rows=torch.randint(0, 500, (n, 4)),
                mask=torch.ones(n, 4, dtype=torch.bool),
                evidence=torch.rand(n, 4, dtype=torch.float32),
            )

        hidden = torch.randn(2, 10, 2048, requires_grad=True)
        output = mod(hidden)
        loss = output.square().mean()
        loss.backward()

        grad = mod.address_projection.weight.grad
        self.assertIsNotNone(grad)
        # At least some elements should have non-trivial gradient
        non_zero = (grad.abs() > 1e-10).sum().item()
        self.assertGreater(non_zero, 0)

    def test_frozen_bank_has_no_gradient(self) -> None:
        """The definition bank is frozen and should not receive gradient."""
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)

        for order in (2, 3):
            rows = mod.address_table.route_count * mod.address_table.alphabet_size**order
            addresses = torch.arange(min(10, rows))
            n = addresses.shape[0]
            mod.address_table.populate(
                order,
                addresses,
                sense_rows=torch.randint(0, 500, (n, 4)),
                mask=torch.ones(n, 4, dtype=torch.bool),
                evidence=torch.ones(n, 4, dtype=torch.float32),
            )

        hidden = torch.randn(1, 6, 2048, requires_grad=True)
        output = mod(hidden)
        output.sum().backward()

        # Bank is not a parameter, so no .grad attribute
        self.assertFalse(isinstance(mod.definition_bank, torch.nn.Parameter))
        self.assertFalse(mod.definition_bank.requires_grad)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class AddressTableIntegrationTests(unittest.TestCase):
    def test_populate_and_lookup_roundtrip(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)

        # Populate order-2 address 0 with sense rows [10, 20, 30, 40]
        mod.address_table.populate(
            2,
            torch.tensor([0]),
            sense_rows=torch.tensor([[10, 20, 30, 40]]),
            mask=torch.tensor([[True, True, True, True]]),
            evidence=torch.tensor([[1.0, 0.5, 0.3, 0.2]]),
        )

        # Lookup address 0
        addresses = torch.zeros(1, 1, 1, dtype=torch.long)  # [B, T, routes]
        valid = torch.ones(1, 1, 1, dtype=torch.bool)
        sense_rows, mask, evidence = mod.address_table.lookup(2, addresses, valid)

        self.assertEqual(tuple(sense_rows.shape), (1, 1, 1, 4))
        self.assertEqual(sense_rows[0, 0, 0].tolist(), [10, 20, 30, 40])
        self.assertTrue(mask[0, 0, 0].all())
        self.assertTrue(torch.allclose(evidence[0, 0, 0], torch.tensor([1.0, 0.5, 0.3, 0.2]), atol=1e-5))

    def test_unpopulated_address_returns_masked_false(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)

        # Lookup an unpopulated address
        addresses = torch.tensor([[[100]]], dtype=torch.long)
        valid = torch.tensor([[[True]]])
        sense_rows, mask, evidence = mod.address_table.lookup(2, addresses, valid)

        self.assertTrue((sense_rows < 0).all())
        self.assertFalse(mask.any())
        self.assertTrue((evidence == 0).all())

    def test_invalid_address_zeroed_by_valid_mask(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)

        addresses = torch.tensor([[[0, 1]]], dtype=torch.long)
        valid = torch.tensor([[[True, False]]])
        sense_rows, mask, evidence = mod.address_table.lookup(2, addresses, valid)

        # Route 1 should be masked out
        self.assertTrue((sense_rows[0, 0, 1] < 0).all())
        self.assertFalse(mask[0, 0, 1].any())


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class DefinitionLnGramStatsTests(unittest.TestCase):
    def test_stats_contain_symbols_addresses_movement(self) -> None:
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        bank = torch.randn(500, 2048, dtype=torch.float32)
        mod.attach_definition_bank(bank)

        for order in (2, 3):
            rows = mod.address_table.route_count * mod.address_table.alphabet_size**order
            addresses = torch.arange(min(10, rows))
            n = addresses.shape[0]
            mod.address_table.populate(
                order,
                addresses,
                sense_rows=torch.randint(0, 500, (n, 4)),
                mask=torch.ones(n, 4, dtype=torch.bool),
                evidence=torch.ones(n, 4, dtype=torch.float32),
            )

        hidden = torch.randn(1, 8, 2048)
        output, stats = mod(hidden, return_stats=True)

        self.assertEqual(tuple(stats.symbols.shape), (1, 8, mod.route_count))
        self.assertIn(2, stats.addresses)
        self.assertIn(3, stats.addresses)
        self.assertEqual(tuple(stats.movement.shape), (1, 8, 2048))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class SignedHarMaxDefinitionFieldTests(unittest.TestCase):
    def _make_fusion(self):
        import torch

        from dendritron.memory_fusion import SparseMemoryFusion

        fusion = SparseMemoryFusion(
            2048,
            memory_width=2048,
            hash_rows_by_order={2: 256, 3: 1024},
            hash_heads=4,
            hash_memory_width=88,
        )
        return fusion

    def test_returns_five_tuple(self) -> None:
        import torch

        fusion = self._make_fusion()
        hidden = torch.randn(1, 4, 2048)
        values = torch.randn(1, 4, 3, 2048)
        mask = torch.ones(1, 4, 3, dtype=torch.bool)

        result = fusion._definition_field(hidden, values, mask, None)
        self.assertEqual(len(result), 5)
        update, weights, distances, sense_rows, signed_coeff = result
        self.assertIsNotNone(update)
        self.assertIsNotNone(weights)
        self.assertIsNotNone(distances)
        self.assertIsNone(sense_rows)
        self.assertIsNotNone(signed_coeff)

    def test_evidence_drives_attraction_vs_repulsion(self) -> None:
        """With asymmetric evidence, the signed coefficient should be positive
        for high-evidence anchors and negative for low-evidence ones."""
        import torch

        fusion = self._make_fusion()
        hidden = torch.zeros(1, 1, 2048)
        # Two anchors: one near, one far
        values = torch.zeros(1, 1, 2, 2048)
        values[0, 0, 0, 0] = 1.0  # anchor 0 at +x
        values[0, 0, 1, 0] = -1.0  # anchor 1 at -x
        mask = torch.ones(1, 1, 2, dtype=torch.bool)
        # High evidence on anchor 0, low on anchor 1
        evidence = torch.tensor([[[10.0, 0.01]]])

        update, weights, distances, _, signed_coeff = fusion._definition_field(
            hidden, values, mask, None, evidence,
        )
        # signed_coeff should be positive for anchor 0, negative for anchor 1
        self.assertGreater(signed_coeff[0, 0, 0].item(), 0.0)
        self.assertLess(signed_coeff[0, 0, 1].item(), 0.0)

    def test_uniform_evidence_defaults_to_attraction(self) -> None:
        """With evidence=None, defaults to uniform 1.0 — all valid anchors
        attract equally (equal y-mass)."""
        import torch

        fusion = self._make_fusion()
        hidden = torch.randn(1, 1, 2048)
        values = torch.randn(1, 1, 3, 2048)
        mask = torch.ones(1, 1, 3, dtype=torch.bool)

        update, weights, distances, _, signed_coeff = fusion._definition_field(
            hidden, values, mask, None, None,
        )
        # With uniform evidence, signed_coeff sums to ~0 (both y and p are
        # probability distributions over the same support)
        total = signed_coeff.sum(dim=-1)
        self.assertTrue(total.abs().max() < 1e-4)

    def test_bank_gather_path_returns_five_tuple(self) -> None:
        import torch

        fusion = self._make_fusion()
        bank = torch.randn(100, 2048, dtype=torch.float32)
        fusion.attach_definition_bank(bank)

        hidden = torch.randn(1, 4, 2048)
        sense_rows = torch.randint(0, 100, (1, 4, 3), dtype=torch.long)
        mask = torch.ones(1, 4, 3, dtype=torch.bool)

        result = fusion._definition_field(hidden, None, mask, sense_rows)
        self.assertEqual(len(result), 5)
        update, weights, distances, sr, signed_coeff = result
        self.assertIsNotNone(update)
        self.assertIsNotNone(signed_coeff)

    def test_empty_definitions_returns_five_nones(self) -> None:
        import torch

        fusion = self._make_fusion()
        hidden = torch.randn(1, 4, 2048)
        result = fusion._definition_field(hidden, None, None, None)
        self.assertEqual(len(result), 5)
        for item in result:
            self.assertIsNone(item)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class MemoryPayloadsEvidenceTests(unittest.TestCase):
    def test_evidence_carried_through_to(self) -> None:
        import torch

        from dendritron.memory_fusion import MemoryPayloads

        evidence = torch.randn(2, 8, 4)
        payloads = MemoryPayloads(definition_evidence=evidence)
        moved = payloads.to("cpu")
        self.assertIsNotNone(moved.definition_evidence)
        self.assertTrue(torch.equal(moved.definition_evidence, evidence))

    def test_evidence_none_by_default(self) -> None:
        from dendritron.memory_fusion import MemoryPayloads

        payloads = MemoryPayloads()
        self.assertIsNone(payloads.definition_evidence)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class ModelPipelineWiringTests(unittest.TestCase):
    def test_model_load_definition_bank_propagates_to_definition_lngram(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        model = DendritronLM(tiny_smoke_config(vocab_size=64))
        bank = torch.randn(100, model.config.memory_width, dtype=torch.float32)
        model.load_definition_bank(bank)

        # Check that definition_lngram modules in the core have the bank
        self.assertIsNotNone(model.core.definition_lngram)
        for module in model.core.definition_lngram:
            self.assertIsNotNone(module.definition_bank)
            self.assertTrue(torch.equal(module.definition_bank, model.definition_bank))

    def test_end_to_end_forward_with_bank(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        config = tiny_smoke_config(vocab_size=64)
        model = DendritronLM(config)
        bank = torch.randn(100, config.memory_width, dtype=torch.float32)
        model.load_definition_bank(bank)

        # Populate some address table entries in the definition lngram modules
        for module in model.core.definition_lngram:
            for order in (2, 3):
                rows = module.address_table.route_count * module.address_table.alphabet_size**order
                addresses = torch.arange(min(10, rows))
                n = addresses.shape[0]
                module.address_table.populate(
                    order,
                    addresses,
                    sense_rows=torch.randint(0, 100, (n, config.senses_per_address)),
                    mask=torch.ones(n, config.senses_per_address, dtype=torch.bool),
                    evidence=torch.ones(n, config.senses_per_address, dtype=torch.float32),
                )

        input_ids = torch.randint(0, 64, (2, 16))
        output = model(input_ids)
        self.assertEqual(tuple(output.logits.shape), (2, 16, 64))

    def test_end_to_end_forward_with_stats(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        config = tiny_smoke_config(vocab_size=64)
        model = DendritronLM(config)
        bank = torch.randn(100, config.memory_width, dtype=torch.float32)
        model.load_definition_bank(bank)

        for module in model.core.definition_lngram:
            for order in (2, 3):
                rows = module.address_table.route_count * module.address_table.alphabet_size**order
                addresses = torch.arange(min(10, rows))
                n = addresses.shape[0]
                module.address_table.populate(
                    order,
                    addresses,
                    sense_rows=torch.randint(0, 100, (n, config.senses_per_address)),
                    mask=torch.ones(n, config.senses_per_address, dtype=torch.bool),
                    evidence=torch.ones(n, config.senses_per_address, dtype=torch.float32),
                )

        input_ids = torch.randint(0, 64, (1, 16))
        output = model(input_ids, return_stats=True)
        self.assertIsNotNone(output.recurrent_stats)
        # At least one visit should have definition_lngram stats
        has_def_stats = any(
            v.definition_lngram is not None
            for v in output.recurrent_stats.visits
        )
        self.assertTrue(has_def_stats)

    def test_definition_bank_not_trainable(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        model = DendritronLM(tiny_smoke_config(vocab_size=64))
        bank = torch.randn(100, model.config.memory_width, dtype=torch.float32)
        model.load_definition_bank(bank)

        # Bank should not appear in parameters
        param_names = [name for name, _ in model.named_parameters()]
        self.assertNotIn("definition_bank", param_names)
        self.assertNotIn("core.definition_lngram.0.definition_bank", param_names)

    def test_j_h_t_distinct_from_joint_to_live(self) -> None:
        """J_h^T (training gradient through W_q) must be distinct from
        joint_to_live (runtime movement map).  Verify they are separate
        parameters in the JointTransferDomain."""
        import torch

        from dendritron.definition_lngram import DefinitionLnGram

        mod = DefinitionLnGram(2048, 2048, bits_per_route=4, orders=(2, 3))
        # joint_to_live is the runtime movement map
        joint_to_live = mod.joint_transfer.joint_to_live
        # address_projection is W_q — the training gradient path
        w_q = mod.address_projection

        # They must be different modules with different weights
        self.assertIsNot(joint_to_live, w_q)
        self.assertFalse(torch.equal(
            joint_to_live.weight if hasattr(joint_to_live, 'weight') else torch.tensor([]),
            w_q.weight,
        ))

        # joint_to_live maps joint -> live (memory_width -> model_width)
        # W_q maps joint -> routing logits (memory_width -> model_width)
        # They serve different purposes: runtime movement vs training gradient
        self.assertEqual(joint_to_live.weight.shape[0], mod.model_width)
        self.assertEqual(joint_to_live.weight.shape[1], mod.memory_width)
        self.assertEqual(w_q.weight.shape[0], mod.model_width)
        self.assertEqual(w_q.weight.shape[1], mod.memory_width)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class MemoryFusionStatsTests(unittest.TestCase):
    def test_stats_include_signed_coeff(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.memory_fusion import MemoryPayloads, SparseMemoryFusion

        config = tiny_smoke_config(vocab_size=64)
        fusion = SparseMemoryFusion(
            config.model_width,
            memory_width=config.memory_width,
            hash_rows_by_order=config.hash_rows_by_order,
            hash_heads=config.hash_heads,
            hash_memory_width=config.hash_memory_width,
        )
        hidden = torch.randn(1, 4, config.model_width)
        definitions = torch.randn(1, 4, 3, config.memory_width)
        mask = torch.ones(1, 4, 3, dtype=torch.bool)
        evidence = torch.rand(1, 4, 3)
        payloads = MemoryPayloads(
            definitions=definitions,
            definition_mask=mask,
            definition_evidence=evidence,
        )

        update, stats = fusion(hidden, payloads, block_index=0, return_stats=True)
        self.assertIsNotNone(stats.definition_signed_coeff)
        self.assertEqual(tuple(stats.definition_signed_coeff.shape), (1, 4, 3))

    def test_stats_signed_coeff_none_when_no_definitions(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.memory_fusion import MemoryPayloads, SparseMemoryFusion

        config = tiny_smoke_config(vocab_size=64)
        fusion = SparseMemoryFusion(
            config.model_width,
            memory_width=config.memory_width,
            hash_rows_by_order=config.hash_rows_by_order,
            hash_heads=config.hash_heads,
            hash_memory_width=config.hash_memory_width,
        )
        hidden = torch.randn(1, 4, config.model_width)
        update, stats = fusion(hidden, None, block_index=0, return_stats=True)
        self.assertIsNone(stats.definition_signed_coeff)


if __name__ == "__main__":
    unittest.main()
