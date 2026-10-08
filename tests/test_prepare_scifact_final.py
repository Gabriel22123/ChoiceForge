import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_scifact_final.py"
SPEC = importlib.util.spec_from_file_location("prepare_scifact_final_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def corpus():
    return [{"doc_id": 7, "title": "A study", "abstract": ["First.", "Second."],
             "structured": False}]


def claim(label):
    return {"id": 3, "claim": "A scientific claim", "evidence": {
        "7": [{"label": label, "sentences": [1]}]}, "cited_doc_ids": [7]}


class PrepareSciFactFinalTest(unittest.TestCase):
    def test_claim_without_evidence_is_valid_but_yields_no_pair(self):
        value = claim("SUPPORT")
        value["evidence"] = {}
        rows, keys = MODULE.prepare_rows(corpus(), [value], 10)
        self.assertEqual(rows, [])
        self.assertEqual(keys, [])

    def test_selection_and_candidate_order_are_label_blind(self):
        support, support_keys = MODULE.prepare_rows(corpus(), [claim("SUPPORT")], 10)
        contradict, contradict_keys = MODULE.prepare_rows(corpus(), [claim("CONTRADICT")], 10)
        self.assertEqual(support_keys, contradict_keys)
        self.assertEqual(support[0]["id"], contradict[0]["id"])
        self.assertEqual(support[0]["request"], contradict[0]["request"])
        self.assertNotEqual(support[0]["label"], contradict[0]["label"])

    def test_rationale_sentence_is_the_only_abstract_text(self):
        rows, _ = MODULE.prepare_rows(corpus(), [claim("SUPPORT")], 10)
        self.assertIn("Second.", rows[0]["request"]["context"])
        self.assertNotIn("First.", rows[0]["request"]["context"])

    def test_conflicting_document_labels_fail(self):
        value = claim("SUPPORT")
        value["evidence"]["7"].append({"label": "CONTRADICT", "sentences": [0]})
        with self.assertRaises(ValueError):
            MODULE.prepare_rows(corpus(), [value], 10)


if __name__ == "__main__":
    unittest.main()
