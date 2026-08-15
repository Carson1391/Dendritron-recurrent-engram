from __future__ import annotations

import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "dendritron"


class ArchitectureContractTests(unittest.TestCase):
    def test_revoked_runtime_symbols_are_absent(self) -> None:
        source = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(PACKAGE_ROOT.glob("*.py"))
        )
        forbidden = (
            "SignedGeometricAttention",
            "SenseResolutionState",
            "ConceptLatchState",
            "definition_candidate_top_k",
            "nn.Bilinear",
            "self.metric =",
            "metric_factor",
            "vocabulary_metric",
        )
        for symbol in forbidden:
            self.assertNotIn(symbol, source)

    def test_context_module_uses_harmax_derivative_contract(self) -> None:
        source = (PACKAGE_ROOT / "geometric_attention.py").read_text(
            encoding="utf-8"
        )
        for required in (
            "class HarMaxContraction",
            "target_mass - distance_mass",
            "signed_coefficients",
            "attraction_mass",
            "repulsion_mass",
            "harmonic_residual",
            "displacement.square().sum(dim=-1)",
        ):
            self.assertIn(required, source)

    def test_definition_reference_frame_is_identity(self) -> None:
        tree = ast.parse(
            (PACKAGE_ROOT / "joint_transfer.py").read_text(encoding="utf-8")
        )
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "definitions_to_joint"
        )
        returns_values = any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Name)
            and node.value.id == "values"
            for node in ast.walk(method)
        )
        self.assertTrue(returns_values)

    def test_two_sequential_residual_sublayers_are_present(self) -> None:
        source = (PACKAGE_ROOT / "recurrent_core.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("contraction_norms", source)
        self.assertIn("compute_norms", source)
        self.assertLess(
            source.index("self.contraction_norms[block_index]"),
            source.index("self.compute_norms[block_index]"),
        )


if __name__ == "__main__":
    unittest.main()
