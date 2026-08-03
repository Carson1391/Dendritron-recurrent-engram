from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stage3_dictionary.definition_sources import (
    DefinitionSourceSpec,
    build_coverage_report,
    build_dictionary_storage_report,
    compile_reviewed_supplement,
    iter_mesh_definitions,
    iter_oewn_definitions,
    iter_wiktionary_definitions,
    parse_wiktionary_snapshot_date,
)


def source(source_id: str, parser: str) -> DefinitionSourceSpec:
    return DefinitionSourceSpec(
        source_id=source_id,
        parser=parser,
        version="fixture",
        filename="fixture",
        url="https://example.test/fixture",
        source_page="https://example.test",
        license_name="fixture license",
        license_url="https://example.test/license",
        attribution="fixture",
    )


class DefinitionSourceTests(unittest.TestCase):
    def test_wiktionary_snapshot_date_is_resolved_from_source_page(self):
        html = (
            "<p>The current version was extracted from the "
            "<a>enwiktionary dump</a> dated 2026-07-06.</p>"
        )
        self.assertEqual(
            parse_wiktionary_snapshot_date(html),
            "2026-07-06",
        )

    def test_oewn_keeps_distinct_single_word_senses(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<LexicalResource xmlns="http://globalwordnet.github.io/schemas/wn">
  <Lexicon id="oewn">
    <LexicalEntry id="bark-n">
      <Lemma writtenForm="bark" partOfSpeech="n"/>
      <Sense id="bark-n-1" synset="synset-bark-tree"/>
    </LexicalEntry>
    <LexicalEntry id="tree-bark-n">
      <Lemma writtenForm="tree bark" partOfSpeech="n"/>
      <Sense id="tree-bark-n-1" synset="synset-bark-tree"/>
    </LexicalEntry>
    <Synset id="synset-bark-tree">
      <Definition>The tough protective outer covering of a tree.</Definition>
      <Example>The bark protected the trunk.</Example>
    </Synset>
  </Lexicon>
</LexicalResource>
"""
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "oewn.xml.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(xml)
            rows = list(iter_oewn_definitions(path, source("oewn", "oewn_lmf")))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["surface"], "bark")
        self.assertEqual(rows[0]["source_sense_key"], "bark-n-1")
        self.assertIn("protective", rows[0]["definition"])
        self.assertEqual(rows[0]["examples"], ["The bark protected the trunk."])

    def test_wiktionary_filters_language_multiword_and_form_rows(self):
        records = [
            {
                "word": "quark",
                "lang_code": "en",
                "pos": "noun",
                "pageid": 12,
                "senses": [
                    {
                        "glosses": [
                            "A fundamental constituent of matter.",
                        ],
                        "topics": ["physics", "particle-physics"],
                        "examples": [{"text": "A proton contains quarks."}],
                    },
                    {
                        "glosses": ["plural of quark"],
                        "tags": ["form-of"],
                    },
                ],
            },
            {
                "word": "campo",
                "lang_code": "es",
                "pos": "noun",
                "senses": [{"glosses": ["field"]}],
            },
            {
                "word": "quantum field",
                "lang_code": "en",
                "pos": "noun",
                "senses": [{"glosses": ["A physical field."]}],
            },
        ]
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "wiktionary.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            rows = list(
                iter_wiktionary_definitions(
                    path,
                    source("wiktionary", "wiktextract_jsonl"),
                )
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["surface"], "quark")
        self.assertEqual(rows[0]["domains"], ["particle-physics", "physics"])
        self.assertEqual(rows[0]["examples"], ["A proton contains quarks."])

    def test_mesh_uses_scope_note_for_each_single_word_term(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<DescriptorRecordSet>
  <DescriptorRecord>
    <DescriptorUI>D000001</DescriptorUI>
    <DescriptorName><String>Calcimycin</String></DescriptorName>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <ConceptUI>M0000001</ConceptUI>
        <ScopeNote>An ionophorous antibiotic that transports calcium.</ScopeNote>
        <TermList>
          <Term><TermUI>T000001</TermUI><String>Calcimycin</String></Term>
          <Term><TermUI>T000002</TermUI><String>Calcium Ionophore</String></Term>
        </TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>
"""
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mesh.xml.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(xml)
            rows = list(
                iter_mesh_definitions(path, source("mesh", "mesh_xml"))
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["surface"], "Calcimycin")
        self.assertEqual(rows[0]["source_sense_key"], "D000001:M0000001:T000001")
        self.assertIn("antibiotic", rows[0]["definition"])

    def test_coverage_is_measured_against_completed_ngram_words(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            definitions = root / "definitions.jsonl"
            definitions.write_text(
                "\n".join(
                    (
                        json.dumps({"surface": "quark"}),
                        json.dumps({"surface": "field"}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            ngrams = root / "keys.jsonl"
            ngrams.write_text(
                "\n".join(
                    (
                        json.dumps({"text": "quantum field", "frequency": 10}),
                        json.dumps({"text": "quark field", "frequency": 5}),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_coverage_report([definitions], [ngrams])
        self.assertEqual(report["unique_ngram_words"], 3)
        self.assertEqual(report["defined_ngram_words"], 2)
        self.assertEqual(report["missing_ngram_words"], 1)
        self.assertEqual(
            report["highest_frequency_missing_words"][0]["word"],
            "quantum",
        )

    def test_storage_report_counts_every_dictionary_sense(self):
        with TemporaryDirectory() as temporary:
            definitions = Path(temporary) / "definitions.jsonl"
            definitions.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "surface": "bark",
                                "definition": "The outer covering of a tree.",
                            }
                        ),
                        json.dumps(
                            {
                                "surface": "bark",
                                "definition": "The sound made by a dog.",
                            }
                        ),
                        json.dumps(
                            {
                                "surface": "photosynthesis",
                                "definition": "Conversion of light into energy.",
                            }
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            report = build_dictionary_storage_report([definitions])
        self.assertEqual(report["sense_rows"], 3)
        self.assertEqual(report["headwords"], 2)
        self.assertEqual(
            report["estimated_definition_vector_bytes_bf16"],
            3 * 2048 * 2,
        )
        self.assertEqual(report["dendritron_cuda_bytes"], 0)

    def test_reviewed_supplement_preserves_multiple_senses(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            supplement = root / "science.jsonl"
            supplement.write_text(
                "\n".join(
                    (
                        json.dumps(
                            {
                                "word": "field",
                                "pos": "noun",
                                "definition": "A region with a physical influence.",
                                "sense_key": "field-physics",
                                "source": "reviewed-science",
                            }
                        ),
                        json.dumps(
                            {
                                "word": "field",
                                "pos": "noun",
                                "definition": "An area of academic study.",
                                "sense_key": "field-study",
                                "source": "reviewed-science",
                            }
                        ),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            result = compile_reviewed_supplement(
                supplement,
                root / "canonical",
            )
            output = Path(result["canonical"]["path"])
            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_sense_key"] for row in rows}, {
            "field-physics",
            "field-study",
        })


if __name__ == "__main__":
    unittest.main()
