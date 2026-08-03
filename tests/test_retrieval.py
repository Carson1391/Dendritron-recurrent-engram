from __future__ import annotations

import unittest

from dendritron.retrieval import LongestEngramRouter, MemoryCandidate


class LongestEngramRouterTests(unittest.TestCase):
    def setUp(self):
        self.trigram = MemoryCandidate(
            kind="donor_engram",
            word_order=3,
            row_index=7,
            surface_text="rough tree bark",
            recipient_ids=(10, 11, 12, 13),
            boundary_mode="bos",
            frequency=50,
        )
        self.bigram = MemoryCandidate(
            kind="donor_engram",
            word_order=2,
            row_index=8,
            surface_text="tree bark",
            recipient_ids=(12, 13),
            boundary_mode="internal",
            frequency=100,
        )
        self.bark_tree = MemoryCandidate(
            kind="dictionary_sense",
            word_order=1,
            row_index=20,
            surface_text="bark",
            recipient_ids=(13,),
            boundary_mode="internal",
            sense_id="bark-tree",
        )
        self.bark_dog = MemoryCandidate(
            kind="dictionary_sense",
            word_order=1,
            row_index=21,
            surface_text="bark",
            recipient_ids=(13,),
            boundary_mode="internal",
            sense_id="bark-dog",
        )

    def test_three_word_match_wins_even_with_four_recipient_tokens(self):
        router = LongestEngramRouter(
            (self.trigram, self.bigram, self.bark_tree, self.bark_dog)
        )
        result = router.resolve((10, 11, 12, 13), 3, include_decomposition=True)
        self.assertIsNotNone(result)
        self.assertEqual(result.word_order, 3)
        self.assertEqual(result.selected, (self.trigram,))
        self.assertEqual(
            {item.word_order for item in result.decomposition_candidates},
            {1, 2},
        )

    def test_bigram_is_used_when_no_trigram_exists(self):
        router = LongestEngramRouter(
            (self.bigram, self.bark_tree, self.bark_dog)
        )
        result = router.resolve((99, 12, 13), 2)
        self.assertIsNotNone(result)
        self.assertEqual(result.word_order, 2)
        self.assertEqual(result.selected, (self.bigram,))

    def test_dictionary_fallback_preserves_all_senses(self):
        router = LongestEngramRouter((self.bark_tree, self.bark_dog))
        result = router.resolve((99, 13), 1)
        self.assertIsNotNone(result)
        self.assertEqual(result.word_order, 1)
        self.assertEqual(
            {item.sense_id for item in result.selected},
            {"bark-tree", "bark-dog"},
        )
        self.assertTrue(result.requires_sense_selection)


if __name__ == "__main__":
    unittest.main()
