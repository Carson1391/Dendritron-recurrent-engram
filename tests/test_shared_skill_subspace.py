from __future__ import annotations

import unittest

import numpy as np

from dendritron.shared_skill_subspace import (
    AdapterCoefficients,
    LoRAFactors,
    fit_shared_skill_basis,
    orthogonal_update_residual,
)


class SharedSkillSubspaceTests(unittest.TestCase):
    def test_shared_task_factor_space_reconstructs_seen_adapters(self):
        random = np.random.default_rng(19)
        beta_basis = random.normal(size=(12, 3))
        alpha_basis = random.normal(size=(10, 3))
        adapters = []
        for index in range(5):
            beta_coordinates = random.normal(size=(3, 2))
            alpha_coordinates = random.normal(size=(3, 2))
            adapters.append(
                LoRAFactors(
                    task_id=f"task-{index}",
                    beta=beta_basis @ beta_coordinates,
                    alpha=(alpha_basis @ alpha_coordinates).T,
                )
            )
        basis, coordinates = fit_shared_skill_basis(adapters, skill_rank=3)
        for adapter, task_coordinates in zip(
            adapters,
            coordinates,
            strict=True,
        ):
            restored = basis.reconstruct_factors(task_coordinates)
            relative_error = np.linalg.norm(restored.delta() - adapter.delta()) / (
                np.linalg.norm(adapter.delta()) + 1e-12
            )
            self.assertLess(float(relative_error), 1e-5)
            self.assertIsInstance(task_coordinates, AdapterCoefficients)

    def test_x_part_is_orthogonal_to_existing_basis(self):
        basis = np.eye(6, 2)
        updates = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
        _, residual = orthogonal_update_residual(basis, updates)
        np.testing.assert_allclose(basis.T @ residual, 0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
