import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("semantic_scale_report", Path(__file__).parents[1] / "scripts/semantic_scale_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class PairedReportTests(unittest.TestCase):
    def test_counts_preserve_pairing_and_shared_groups(self):
        rows = [{"id": str(i), "source": "snli", "group_id": "same-premise", "label": "yes",
                 "request": {"choices": [{"id": "yes"}, {"id": "no"}]}} for i in range(4)]
        def prediction(i, correct):
            return {"id": str(i), "label": "yes", "probabilities": {"yes": .9 if correct else .1, "no": .1 if correct else .9}}
        initial = [prediction(i, i < 2) for i in range(4)]
        final = [prediction(i, i % 2 == 0) for i in range(4)]
        result = report.paired_diagnostics(rows, initial, final)["snli"]
        self.assertEqual([result[k] for k in ("fixed", "regressed", "both_correct", "both_wrong")], [1, 1, 1, 1])
        self.assertEqual(result["groups"], 1)
        self.assertEqual(result["paired_group_bootstrap_95_percentile"], [0, 0])
        self.assertEqual(result["confusion"]["trained"]["yes"], {"yes": 2, "no": 2})

    def test_rejects_unpaired_predictions(self):
        with self.assertRaisesRegex(ValueError, "IDs"):
            report.paired_diagnostics([], [{"id": "extra"}], [])
