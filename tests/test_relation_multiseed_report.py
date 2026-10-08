from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from relation_group_multiseed_report import neutral_stats


class RelationMultiseedReportTests(unittest.TestCase):
    def test_neutral_recall_and_precision_are_kept_distinct(self):
        matrix = {
            "contradiction": {"contradiction": 5, "neutral": 2},
            "neutral": {"neutral": 3, "entailment": 1},
            "entailment": {"entailment": 6, "neutral": 1},
        }
        result = {"arms": {"grouped": {"paired_vs_initial": {"snli": {
            "confusion": {"trained": matrix}}}}}}
        value = neutral_stats(result, "grouped")
        self.assertEqual(value["correct"], 3)
        self.assertEqual(value["total"], 4)
        self.assertEqual(value["predicted"], 6)
        self.assertEqual(value["recall"], .75)
        self.assertEqual(value["precision"], .5)
        self.assertEqual(value["per_class_correct"], {
            "contradiction": 5, "neutral": 3, "entailment": 6})


if __name__ == "__main__":
    unittest.main()
