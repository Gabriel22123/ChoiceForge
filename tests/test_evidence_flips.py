import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from evidence_flip_probe import cases
from decision_model.core import validate_request


class EvidenceFlipTests(unittest.TestCase):
    def test_only_premise_changes_inside_each_balanced_group(self):
        examples = cases()
        self.assertEqual(len(examples), 18)
        self.assertEqual(len({c["id"] for c in examples}), 18)
        for group in {c["group"] for c in examples}:
            selected = [c for c in examples if c["group"] == group]
            self.assertEqual({c["expected"] for c in selected}, {"entailment", "neutral", "contradiction"})
            self.assertEqual(len({c["request"]["task"] for c in selected}), 1)
            self.assertEqual(len({json.loads(c["request"]["context"])["hypothesis"] for c in selected}), 1)
            self.assertEqual(len({json.loads(c["request"]["context"])["premise"] for c in selected}), 3)
            for case in selected:
                validate_request(case["request"])
                self.assertEqual(case["request"]["choices"], selected[0]["request"]["choices"])
