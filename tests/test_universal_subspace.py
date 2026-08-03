from __future__ import annotations

import unittest

import numpy as np

from dendritron.universal_subspace import (
    accumulate_source_moments,
    fit_universal_knowledge_basis,
    reconstruction_cosine,
    select_knowledge_rank,
)


class UniversalSubspaceTests(unittest.TestCase):
    def test_known_low_rank_dictionary_geometry_is_recovered(self):
        random = np.random.default_rng(11)
        latent = random.normal(size=(1000, 3))
        transform = random.normal(size=(3, 16))
        vectors = latent @ transform
        moments = accumulate_source_moments(
            "definitions-layer02",
            [vectors[:400], vectors[400:]],
            width=16,
            l2_normalize=True,
        )
        basis = fit_universal_knowledge_basis([moments], rank=3)
        cosine = reconstruction_cosine(basis, vectors[:100])
        self.assertGreater(float(cosine.mean()), 0.99)
        self.assertEqual(basis.rank, 3)

    def test_rank_16_is_kept_only_when_it_reaches_target(self):
        ratio = np.zeros(64, dtype=np.float64)
        ratio[:16] = 0.98 / 16
        ratio[16:] = 0.02 / 48
        selection = select_knowledge_rank(ratio, candidates=(16, 32, 64))
        self.assertEqual(selection.rank, 16)
        self.assertTrue(selection.target_reached)

    def test_rank_expands_when_16_directions_fall_short(self):
        ratio = np.zeros(64, dtype=np.float64)
        ratio[:16] = 0.80 / 16
        ratio[16:32] = 0.19 / 16
        ratio[32:] = 0.01 / 32
        selection = select_knowledge_rank(ratio, candidates=(16, 32, 64))
        self.assertEqual(selection.rank, 32)
        self.assertTrue(selection.target_reached)


if __name__ == "__main__":
    unittest.main()
