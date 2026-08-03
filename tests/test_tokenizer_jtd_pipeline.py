from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from dendritron.hash_engram import (
    HashEngramAddressor,
    hash_ngram_ids,
)
from dendritron.jtd import (
    SurfaceMemoryIndex,
    JTDSourceRecord,
    compile_surface_index_database,
)
from dendritron.memory_pipeline import SurfaceMemoryAddressor
from dendritron.tokenizer import (
    LOCKED_QWEN_TOKENIZER_ID,
    boundary_token_ids,
    build_canonical_token_projection,
    build_tokenizer_contract,
    canonical_token_text,
    complete_word_segments,
)


class FakeQwenTokenizer:
    vocab_size = 128
    is_fast = True
    padding_side = "right"
    truncation_side = "right"
    bos_token_id = None
    eos_token_id = 127
    pad_token_id = 127
    unk_token_id = 0
    mask_token_id = None

    _decoded = {
        0: "<unk>",
        1: "rough",
        2: "tree",
        3: "bark",
        4: "dog",
        5: " ",
        6: "Tree",
        7: ",",
        8: ".",
        9: "café",
        10: "cafe",
        101: " rough",
        102: " tree",
        103: " bark",
        104: " dog",
        105: " bark,",
        106: "tree,",
        107: "\t",
    }

    _encodings = {
        "bark": [3],
        " bark": [103],
        "tree bark": [2, 103],
        " tree bark": [102, 103],
        "rough tree bark": [1, 102, 103],
        " rough tree bark": [101, 102, 103],
        "tree bark,": [2, 105],
        "tree, bark": [106, 103],
    }

    _offsets = {
        "tree bark,": [(0, 4), (4, 10)],
        "tree, bark": [(0, 5), (5, 10)],
    }

    def __len__(self):
        return self.vocab_size

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("JTD must disable special tokens")
        return list(self._encodings[text])

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(
            self._decoded.get(int(token_id), f"<{int(token_id)}>")
            for token_id in token_ids
        )

    def convert_ids_to_tokens(self, token_id):
        return self._decoded.get(int(token_id), f"tok-{int(token_id)}")

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        return_offsets_mapping=False,
        truncation=False,
    ):
        del truncation
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        result = {"input_ids": ids}
        if return_offsets_mapping:
            offsets = self._offsets.get(text)
            if offsets is None:
                raise AssertionError(f"No synthetic offsets for {text!r}")
            result["offset_mapping"] = offsets
        return result


def fake_contract(tokenizer):
    return build_tokenizer_contract(
        tokenizer,
        tokenizer_id=LOCKED_QWEN_TOKENIZER_ID,
        requested_revision="main",
        resolved_revision="0123456789abcdef",
    )


def fake_projection(tokenizer):
    return build_canonical_token_projection(tokenizer)


class TokenizerJTDPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeQwenTokenizer()

    def test_boundary_compilation_matches_full_sequence_behavior(self):
        variants = boundary_token_ids(self.tokenizer, "tree bark")
        self.assertEqual(variants["bos"], (2, 103))
        self.assertEqual(variants["internal"], (102, 103))
        self.assertEqual(
            tuple(self.tokenizer.encode("rough tree bark")),
            (1, 102, 103),
        )

    def test_engram_projection_matches_official_normalization(self):
        projection = fake_projection(self.tokenizer)
        self.assertEqual(projection.project((2,)), projection.project((6,)))
        self.assertEqual(projection.project((2,)), projection.project((102,)))
        self.assertEqual(projection.project((9,)), projection.project((10,)))
        self.assertEqual(projection.project((5,)), projection.project((107,)))
        self.assertNotEqual(projection.project((7,)), projection.project((8,)))
        self.assertEqual(canonical_token_text("  CAFÉ  "), "cafe")

    def test_complete_word_segments_make_punctuation_a_boundary(self):
        self.assertEqual(
            complete_word_segments("tree bark. dog bark"),
            (("tree", "bark"), ("dog", "bark")),
        )
        self.assertEqual(
            complete_word_segments("camera-based don't"),
            (("camera-based", "don't"),),
        )

    def test_jtd_preserves_longest_match_and_dictionary_polysemy(self):
        records = [
            JTDSourceRecord(
                bank_name="bigrams",
                word_order=2,
                row_index=8,
                surface_text="tree bark",
                frequency=100,
            ),
            JTDSourceRecord(
                bank_name="trigrams",
                word_order=3,
                row_index=7,
                surface_text="rough tree bark",
                frequency=50,
            ),
            JTDSourceRecord(
                bank_name="dictionary",
                word_order=1,
                row_index=20,
                surface_text="bark",
                sense_id="bark-tree",
            ),
            JTDSourceRecord(
                bank_name="dictionary",
                word_order=1,
                row_index=21,
                surface_text="bark",
                sense_id="bark-dog",
            ),
        ]
        contract = fake_contract(self.tokenizer)
        projection = fake_projection(self.tokenizer)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = compile_surface_index_database(
                records,
                tokenizer=self.tokenizer,
                tokenizer_contract=contract,
                token_projection=projection,
                database_path=root / "surface_index.sqlite3",
                collision_report_path=root / "collisions.jsonl",
            )
            self.assertEqual(report["source_rows"]["trigrams"], 1)
            self.assertEqual(
                report["collision_report"]["cryptographic_hash_collisions"],
                0,
            )
            with SurfaceMemoryIndex(
                root / "surface_index.sqlite3",
                expected_tokenizer_fingerprint=contract.fingerprint,
            ) as index:
                trigram = index.resolve((1, 102, 103), 2)
                self.assertEqual(trigram.word_order, 3)
                self.assertEqual(trigram.selected[0].row_index, 7)

                bigram = index.resolve((4, 102, 103), 2)
                self.assertEqual(bigram.word_order, 2)
                self.assertEqual(bigram.selected[0].row_index, 8)

                dictionary = index.resolve((4, 103), 1)
                self.assertEqual(dictionary.word_order, 1)
                self.assertEqual(
                    {item.sense_id for item in dictionary.selected},
                    {"bark-tree", "bark-dog"},
                )

    def test_hash_engram_runs_on_donor_miss_alongside_dictionary(self):
        contract = fake_contract(self.tokenizer)
        projection = fake_projection(self.tokenizer)
        records = [
            JTDSourceRecord(
                bank_name="bigrams",
                word_order=2,
                row_index=0,
                surface_text="tree bark",
                frequency=10,
            ),
            JTDSourceRecord(
                bank_name="dictionary",
                word_order=1,
                row_index=0,
                surface_text="bark",
                sense_id="bark-tree",
            ),
        ]
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            compile_surface_index_database(
                records,
                tokenizer=self.tokenizer,
                tokenizer_contract=contract,
                token_projection=projection,
                database_path=root / "surface_index.sqlite3",
                collision_report_path=root / "collisions.jsonl",
            )
            with SurfaceMemoryIndex(root / "surface_index.sqlite3") as index:
                addressor = SurfaceMemoryAddressor(
                    index,
                    hash_engram=HashEngramAddressor(
                        heads=2,
                        table_rows={2: 101, 3: 103},
                    ),
                )
                donor_hit = addressor.resolve((4, 102, 103), 2)
                self.assertTrue(donor_hit.has_frozen_donor_hit)
                self.assertIsNone(donor_hit.hash_engram)

                dictionary_hit = addressor.resolve((4, 103), 1)
                self.assertTrue(dictionary_hit.has_dictionary_candidates)
                self.assertIsNotNone(dictionary_hit.hash_engram)
                self.assertIn(2, dictionary_hit.hash_engram.by_order)

                fused = addressor.resolve_text(self.tokenizer, "tree bark,")
                self.assertTrue(fused[1].has_frozen_donor_hit)
                self.assertEqual(fused[1].exact_memory.word_order, 2)

                separated = addressor.resolve_text(self.tokenizer, "tree, bark")
                self.assertFalse(separated[1].has_frozen_donor_hit)
                self.assertTrue(separated[1].has_dictionary_candidates)
                self.assertIsNotNone(separated[1].hash_engram)

    def test_hash_engram_is_deterministic_and_head_specific(self):
        first = hash_ngram_ids(
            (12, 13),
            order=2,
            head=0,
            table_rows=10_007,
        )
        repeat = hash_ngram_ids(
            (12, 13),
            order=2,
            head=0,
            table_rows=10_007,
        )
        second_head = hash_ngram_ids(
            (12, 13),
            order=2,
            head=1,
            table_rows=10_007,
        )
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, second_head)

    def test_jtd_rejects_tokenizer_fingerprint_mismatch(self):
        contract = fake_contract(self.tokenizer)
        projection = fake_projection(self.tokenizer)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            compile_surface_index_database(
                [
                    JTDSourceRecord(
                        bank_name="dictionary",
                        word_order=1,
                        row_index=0,
                        surface_text="bark",
                        sense_id="bark-tree",
                    )
                ],
                tokenizer=self.tokenizer,
                tokenizer_contract=contract,
                token_projection=projection,
                database_path=root / "surface_index.sqlite3",
                collision_report_path=root / "collisions.jsonl",
            )
            with self.assertRaises(ValueError):
                SurfaceMemoryIndex(
                    root / "surface_index.sqlite3",
                    expected_tokenizer_fingerprint="different",
                )


if __name__ == "__main__":
    unittest.main()
