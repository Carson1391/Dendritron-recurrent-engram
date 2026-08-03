from __future__ import annotations

import unittest
import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from dendritron.definition_bank import (
    DEFINITION_READOUT_MARKER,
    build_dictionary_records,
    canonical_definition_text,
)
from stage3_dictionary.build_dictionary_inventory import (
    build_inventory,
    build_lookup_database,
    file_sha256,
)


class DefinitionBankTests(unittest.TestCase):
    def test_polysemous_word_keeps_separate_sense_rows(self):
        words, senses, report = build_dictionary_records(
            [
                {
                    "word": "bark",
                    "pos": "noun",
                    "definition": "The tough protective outer covering of a tree.",
                    "sense_key": "bark-tree",
                },
                {
                    "word": "bark",
                    "pos": "verb",
                    "definition": "To make the sharp explosive sound of a dog.",
                    "sense_key": "bark-dog",
                },
            ],
            ngram_words=["rough", "tree", "bark"],
        )
        bark_word = next(word for word in words if word.normalized == "bark")
        bark_senses = [sense for sense in senses if sense.word_id == bark_word.word_id]
        self.assertEqual(len(bark_senses), 2)
        self.assertNotEqual(bark_senses[0].sense_id, bark_senses[1].sense_id)
        all_definition_words = {
            word for sense in bark_senses for word in sense.definition_words
        }
        self.assertIn("protective", all_definition_words)
        self.assertIn("sound", all_definition_words)
        self.assertEqual(report["definition_vector_layer"], 2)

    def test_definition_constituents_are_promoted_to_word_rows(self):
        words, senses, _ = build_dictionary_records(
            [
                {
                    "word": "tree",
                    "definition": "A woody perennial plant.",
                    "sense_key": "tree-1",
                }
            ]
        )
        normalized = {word.normalized for word in words}
        self.assertTrue({"tree", "woody", "perennial", "plant"} <= normalized)
        self.assertEqual(
            senses[0].definition_words,
            ("A", "woody", "perennial", "plant"),
        )

    def test_dictionary_senses_extend_beyond_engram_vocabulary(self):
        words, senses, report = build_dictionary_records(
            [
                {
                    "word": "bark",
                    "definition": "The outer covering of a tree.",
                    "sense_key": "bark-tree",
                },
                {
                    "word": "photosynthesis",
                    "definition": "A process that converts light into energy.",
                    "sense_key": "photosynthesis-1",
                },
            ],
            ngram_words=["bark"],
        )
        sense_words = {
            next(word.normalized for word in words if word.word_id == sense.word_id)
            for sense in senses
        }
        self.assertEqual(sense_words, {"bark", "photosynthesis"})
        self.assertEqual(report["ngram_words_without_definition"], 0)

    def test_canonical_marker_is_identical(self):
        first = canonical_definition_text("First meaning.")
        second = canonical_definition_text("Second meaning.")
        self.assertTrue(first.endswith(DEFINITION_READOUT_MARKER))
        self.assertTrue(second.endswith(DEFINITION_READOUT_MARKER))

    def test_sqlite_graph_resolves_definition_word_ids(self):
        words, senses, _ = build_dictionary_records(
            [
                {
                    "word": "bark",
                    "definition": "A protective tree covering.",
                    "sense_key": "bark-tree",
                }
            ]
        )
        with TemporaryDirectory() as temporary:
            database = Path(temporary) / "dictionary.sqlite3"
            counts = build_lookup_database(database, words, senses)
            self.assertTrue(database.is_file())
            self.assertEqual(
                counts["definition_word_edges"],
                len(senses[0].definition_words),
            )

    def test_inventory_manifest_fingerprints_definition_sources(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "science.jsonl"
            definitions.write_text(
                json.dumps(
                    {
                        "word": "quark",
                        "definition": "An elementary particle.",
                        "sense_key": "quark-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ngrams = root / "keys.jsonl"
            ngrams.write_text(
                json.dumps({"text": "elementary quark"}) + "\n",
                encoding="utf-8",
            )
            report = build_inventory(
                Namespace(
                    wordnet=False,
                    download_wordnet=False,
                    definitions=[definitions],
                    ngram_keys=[ngrams],
                    output=root / "inventory",
                )
            )
            self.assertEqual(
                report["input_definition_artifacts"][0]["path"],
                str(definitions),
            )
            self.assertEqual(
                len(report["input_definition_artifacts"][0]["sha256"]),
                64,
            )

    def test_inventory_consumes_verified_canonical_source_manifest(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "oewn.jsonl"
            definitions.write_text(
                json.dumps(
                    {
                        "word": "quark",
                        "definition": "An elementary particle.",
                        "source": "oewn:fixture",
                        "sense_key": "quark-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            ngrams = root / "keys.jsonl"
            ngrams.write_text(
                json.dumps({"text": "elementary quark"}) + "\n",
                encoding="utf-8",
            )
            source_manifest = root / "definition_sources_manifest.json"
            source_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "parser_version": 2,
                        "selection_policy": {
                            "headword_units": 1,
                            "payload_scope": (
                                "complete_curated_single_word_dictionary"
                            ),
                            "definition_word_policy": "ordered_word_id_links",
                        },
                        "coverage": {
                            "type_coverage": 1.0,
                            "missing_ngram_words": 0,
                        },
                        "canonical_definition_files": [
                            {
                                "path": str(definitions),
                                "sha256": file_sha256(definitions),
                            }
                        ],
                        "ngram_key_files": [
                            {
                                "path": str(ngrams),
                                "sha256": file_sha256(ngrams),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_inventory(
                Namespace(
                    definition_manifest=source_manifest,
                    wordnet=False,
                    download_wordnet=False,
                    definitions=[],
                    ngram_keys=[ngrams],
                    output=root / "inventory",
                )
            )
        self.assertEqual(report["sense_rows"], 1)
        self.assertEqual(
            report["definition_source_manifest"]["coverage"]["type_coverage"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
