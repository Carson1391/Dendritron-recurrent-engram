from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from word_unit_table import WordUnitTable


class FakeFastTokenizer:
    """Character tokenizer with the small Hugging Face surface used by the tests."""

    name_or_path = "fake-character-tokenizer"
    all_special_ids = [0]
    is_fast = True

    def __init__(self) -> None:
        characters = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
            " -'’.,"
            "é"
        )
        self._vocab = {"<pad>": 0}
        for character in characters:
            if character not in self._vocab:
                self._vocab[character] = len(self._vocab)
        self._inverse = {value: key for key, value in self._vocab.items()}

    def get_vocab(self):
        return dict(self._vocab)

    def convert_ids_to_tokens(self, token_id):
        return self._inverse[int(token_id)]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del clean_up_tokenization_spaces
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in self.all_special_ids:
                continue
            pieces.append(self._inverse[token_id])
        return "".join(pieces)

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
        truncation=False,
    ):
        del add_special_tokens, truncation
        if isinstance(text, list):
            raise TypeError("The fake tokenizer expects one string")
        ids = [self._vocab[character] for character in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


class WordUnitTableTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "words.sqlite3"
        self.tokenizer = FakeFastTokenizer()

    def tearDown(self):
        self.temporary.cleanup()

    def test_every_input_word_is_appended(self):
        with WordUnitTable(self.database, self.tokenizer) as table:
            encoding = table.encode_input("camera, human food")
            self.assertEqual(
                [word.surface for word in encoding.words],
                ["camera", "human", "food"],
            )
            self.assertEqual(table.counts()["words"], 3)

            table.encode_input("camera photosynthesis")
            self.assertEqual(table.counts()["words"], 4)
            self.assertIsNotNone(table.inspect_word("photosynthesis"))

    def test_exact_surface_and_normalized_group_are_both_preserved(self):
        with WordUnitTable(self.database, self.tokenizer) as table:
            table.encode_input("Camera camera")
            upper = table.inspect_word("Camera")
            lower = table.inspect_word("camera")
            self.assertIsNotNone(upper)
            self.assertIsNotNone(lower)
            assert upper is not None
            assert lower is not None
            self.assertNotEqual(upper["word_id"], lower["word_id"])
            self.assertEqual(upper["normalized"], "camera")
            self.assertEqual(lower["normalized"], "camera")

    def test_token_to_word_alignment_keeps_punctuation_unassigned(self):
        with WordUnitTable(self.database, self.tokenizer) as table:
            encoding = table.encode_input("food.")
            punctuation_position = len("food")
            self.assertEqual(encoding.token_to_word_ids[punctuation_position], ())
            self.assertTrue(
                all(
                    encoding.token_to_word_ids[index]
                    for index in range(len("food"))
                )
            )

    def test_hyphenated_and_apostrophe_words_are_single_units(self):
        with WordUnitTable(self.database, self.tokenizer) as table:
            encoding = table.encode_input("camera-based it's")
            self.assertEqual(
                [word.surface for word in encoding.words],
                ["camera-based", "it's"],
            )


if __name__ == "__main__":
    unittest.main()
