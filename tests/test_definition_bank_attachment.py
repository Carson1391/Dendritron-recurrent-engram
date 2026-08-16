"""Tests for frozen definition bank attachment and LNGram address -> sense gather.

Verifies:
1. DendritronLM.load_definition_bank registers a non-trainable buffer
2. Double-load is rejected
3. Shape validation enforces [N_senses, memory_width]
4. SparseMemoryFusion.attach_definition_bank stores the bank
5. _definition_field gathers from the bank when values=None but sense_rows+mask given
6. Invalid sense rows (-1) are masked out
7. Bank-gathered vectors match explicit definition tensors
8. End-to-end forward pass works with bank-gathered definitions
9. Bank buffer is not a trainable parameter
"""

from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch executes on the training host")
class DefinitionBankAttachmentTests(unittest.TestCase):
    def test_load_definition_bank_registers_non_trainable_buffer(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        model = DendritronLM(tiny_smoke_config(vocab_size=64))
        bank = torch.randn(100, model.config.memory_width, dtype=torch.float32)
        model.load_definition_bank(bank)

        self.assertTrue(hasattr(model, "definition_bank"))
        self.assertIsInstance(model.definition_bank, torch.Tensor)
        self.assertEqual(tuple(model.definition_bank.shape), (100, model.config.memory_width))
        self.assertFalse(model.definition_bank.requires_grad)

    def test_double_load_raises(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        model = DendritronLM(tiny_smoke_config(vocab_size=64))
        bank = torch.randn(50, model.config.memory_width)
        model.load_definition_bank(bank)
        with self.assertRaises(RuntimeError):
            model.load_definition_bank(bank)

    def test_shape_validation_rejects_wrong_dims(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        model = DendritronLM(tiny_smoke_config(vocab_size=64))
        bad_bank = torch.randn(50, model.config.memory_width + 1)
        with self.assertRaises(ValueError):
            model.load_definition_bank(bad_bank)

        bad_bank_3d = torch.randn(10, 50, model.config.memory_width)
        with self.assertRaises(ValueError):
            model.load_definition_bank(bad_bank_3d)

    def test_attach_definition_bank_stores_in_fusion(self) -> None:
        import torch

        from dendritron.memory_fusion import SparseMemoryFusion

        fusion = SparseMemoryFusion(
            8,
            memory_width=8,
            hash_rows_by_order={2: 16, 3: 32},
            hash_heads=2,
            hash_memory_width=4,
        )
        self.assertIsNone(fusion.definition_bank)
        bank = torch.randn(40, 8)
        fusion.attach_definition_bank(bank)
        self.assertIsNotNone(fusion.definition_bank)
        self.assertTrue(torch.equal(fusion.definition_bank, bank))

    def test_definition_field_gathers_from_bank(self) -> None:
        import torch

        from dendritron.memory_fusion import SparseMemoryFusion

        width = 8
        n_senses = 50
        fusion = SparseMemoryFusion(
            width,
            memory_width=width,
            hash_rows_by_order={2: 16, 3: 32},
            hash_heads=2,
            hash_memory_width=4,
        )
        bank = torch.randn(n_senses, width)
        fusion.attach_definition_bank(bank)

        hidden = torch.randn(2, 3, width)
        sense_rows = torch.tensor([[[0, 1, 2], [10, 11, 12], [20, 21, 22]],
                                   [[5, 6, 7], [15, 16, 17], [25, 26, 27]]])
        mask = torch.ones(2, 3, 3, dtype=torch.bool)

        update, weights, distances, returned_rows, signed_coeff = fusion._definition_field(
            hidden, None, mask, sense_rows
        )
        self.assertIsNotNone(update)
        self.assertEqual(tuple(update.shape), tuple(hidden.shape))
        self.assertTrue((weights > 0).all())
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(2, 3)))

    def test_invalid_sense_rows_are_masked(self) -> None:
        import torch

        from dendritron.memory_fusion import SparseMemoryFusion

        width = 8
        n_senses = 20
        fusion = SparseMemoryFusion(
            width,
            memory_width=width,
            hash_rows_by_order={2: 16, 3: 32},
            hash_heads=2,
            hash_memory_width=4,
        )
        bank = torch.randn(n_senses, width)
        fusion.attach_definition_bank(bank)

        hidden = torch.randn(1, 2, width)
        sense_rows = torch.tensor([[[0, -1, 2], [5, 6, -1]]])
        mask = torch.ones(1, 2, 3, dtype=torch.bool)

        update, weights, distances, returned_rows, signed_coeff = fusion._definition_field(
            hidden, None, mask, sense_rows
        )
        self.assertIsNotNone(update)
        # Positions with -1 should have zero weight
        self.assertEqual(float(weights[0, 0, 1]), 0.0)
        self.assertEqual(float(weights[0, 1, 2]), 0.0)
        # Valid positions should have positive weight
        self.assertGreater(float(weights[0, 0, 0]), 0.0)
        self.assertGreater(float(weights[0, 0, 2]), 0.0)
        self.assertGreater(float(weights[0, 1, 0]), 0.0)
        self.assertGreater(float(weights[0, 1, 1]), 0.0)
        # Returned rows should have -1 for invalid slots
        self.assertEqual(int(returned_rows[0, 0, 1]), -1)
        self.assertEqual(int(returned_rows[0, 1, 2]), -1)

    def test_bank_gather_matches_explicit_definitions(self) -> None:
        import torch

        from dendritron.memory_fusion import SparseMemoryFusion

        width = 8
        n_senses = 30
        fusion = SparseMemoryFusion(
            width,
            memory_width=width,
            hash_rows_by_order={2: 16, 3: 32},
            hash_heads=2,
            hash_memory_width=4,
        )
        bank = torch.randn(n_senses, width)
        fusion.attach_definition_bank(bank)

        hidden = torch.randn(1, 2, width)
        sense_rows = torch.tensor([[[3, 7, 15], [0, 1, 2]]])
        mask = torch.ones(1, 2, 3, dtype=torch.bool)

        # Path A: bank gather (values=None)
        update_bank, w_bank, d_bank, r_bank, sc_bank = fusion._definition_field(
            hidden, None, mask, sense_rows
        )

        # Path B: explicit definitions tensor materialized from the same rows
        explicit_defs = bank[sense_rows.reshape(-1)].view(1, 2, 3, width)
        update_explicit, w_explicit, d_explicit, r_explicit, sc_explicit = fusion._definition_field(
            hidden, explicit_defs, mask, sense_rows
        )

        self.assertTrue(torch.allclose(update_bank, update_explicit, atol=1e-5))
        self.assertTrue(torch.allclose(w_bank, w_explicit, atol=1e-5))
        self.assertTrue(torch.allclose(d_bank, d_explicit, atol=1e-5))

    def test_end_to_end_forward_with_bank_gathered_definitions(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.memory_fusion import MemoryPayloads
        from dendritron.model import DendritronLM

        torch.manual_seed(42)
        model = DendritronLM(tiny_smoke_config(vocab_size=64))
        n_senses = 200
        bank = torch.randn(n_senses, model.config.memory_width)
        model.load_definition_bank(bank)

        token_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        batch, length = token_ids.shape
        max_senses = 3

        # Payloads with only sense_rows + mask, no pre-materialized definitions
        payloads = MemoryPayloads(
            definition_sense_rows=torch.randint(
                0, n_senses, (batch, length, max_senses), dtype=torch.long
            ),
            definition_mask=torch.ones(batch, length, max_senses, dtype=torch.bool),
        )

        output = model(token_ids, memory_payloads=payloads)
        self.assertEqual(tuple(output.logits.shape), (batch, length, 64))
        self.assertTrue(torch.isfinite(output.logits).all())

    def test_bank_gathered_forward_matches_explicit_forward(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.memory_fusion import MemoryPayloads
        from dendritron.model import DendritronLM

        torch.manual_seed(99)
        model = DendritronLM(tiny_smoke_config(vocab_size=64))
        n_senses = 200
        bank = torch.randn(n_senses, model.config.memory_width)
        model.load_definition_bank(bank)

        # Open the definition gates so the definition path actually contributes
        with torch.no_grad():
            model.memory_fusion.definition_gates.fill_(0.7)

        token_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
        batch, length = token_ids.shape
        max_senses = 2
        sense_rows = torch.randint(0, n_senses, (batch, length, max_senses), dtype=torch.long)
        mask = torch.ones(batch, length, max_senses, dtype=torch.bool)

        # Path A: bank-gathered (definitions=None)
        payloads_bank = MemoryPayloads(
            definition_sense_rows=sense_rows,
            definition_mask=mask,
        )
        output_bank = model(token_ids, memory_payloads=payloads_bank)

        # Path B: explicit definitions materialized from same rows
        explicit_defs = bank[sense_rows.reshape(-1)].view(batch, length, max_senses, -1)
        payloads_explicit = MemoryPayloads(
            definitions=explicit_defs,
            definition_mask=mask,
            definition_sense_rows=sense_rows,
        )
        output_explicit = model(token_ids, memory_payloads=payloads_explicit)

        self.assertTrue(torch.allclose(output_bank.logits, output_explicit.logits, atol=1e-4))

    def test_definition_bank_not_in_parameters(self) -> None:
        import torch

        from dendritron.config import tiny_smoke_config
        from dendritron.model import DendritronLM

        model = DendritronLM(tiny_smoke_config(vocab_size=64))
        bank = torch.randn(50, model.config.memory_width)
        model.load_definition_bank(bank)

        param_names = {name for name, _ in model.named_parameters()}
        self.assertNotIn("definition_bank", param_names)

        buffer_names = {name for name, _ in model.named_buffers()}
        self.assertIn("definition_bank", buffer_names)

    def test_no_bank_no_sense_rows_returns_none(self) -> None:
        import torch

        from dendritron.memory_fusion import SparseMemoryFusion

        fusion = SparseMemoryFusion(
            8,
            memory_width=8,
            hash_rows_by_order={2: 16, 3: 32},
            hash_heads=2,
            hash_memory_width=4,
        )
        # No bank attached, no values -> should return None
        hidden = torch.randn(1, 2, 8)
        update, w, d, r, sc = fusion._definition_field(hidden, None, None, None)
        self.assertIsNone(update)


if __name__ == "__main__":
    unittest.main()
