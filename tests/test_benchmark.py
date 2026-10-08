import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from decision_model.benchmark import benchmark
from decision_model.core import digest, load_rows


def _row(case_id, family, label, soft=False):
    row = {
        "id": case_id,
        "split": "test",
        "group_id": case_id,
        "source": "public-source",
        "family": family,
        "evaluation_regime": "held_out_task",
        "label": label,
        "provenance": [{"url": "https://example.com/public"}],
        "request": {
            "task": "Choose the supported answer.",
            "context": f"Public context {case_id}.",
            "choices": [
                {"id": "yes", "description": "The answer is yes."},
                {"id": "no", "description": "The answer is no."},
            ],
        },
    }
    if soft:
        row["target_probabilities"] = (
            {"yes": 0.8, "no": 0.2} if label == "yes" else {"yes": 0.2, "no": 0.8}
        )
    return row


class BenchmarkTests(unittest.TestCase):
    def test_architecture_benchmark_data_is_tracked_and_public(self):
        root = Path(__file__).resolve().parents[1]
        data = root / "data/architecture-benchmark-v1/cases.jsonl"
        manifest = json.loads((data.parent / "manifest.json").read_text())
        self.assertEqual(digest(data.read_bytes()), manifest["dataset_sha256"])
        rows = load_rows(data)
        self.assertEqual(len(rows), 1888)
        self.assertEqual({row["split"] for row in rows}, {"train", "validation", "test"})

    def test_benchmark_reports_calibration_and_selective_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            predictions = root / "predictions.json"
            output = root / "report.json"
            rows = [_row("one", "knowledge", "yes", soft=True),
                    _row("two", "logic", "no", soft=True)]
            data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            predictions.write_text(json.dumps({"predictions": [
                {"id": "one", "choice_id": "yes", "probabilities": {"yes": 0.8, "no": 0.2},
                 "requires_review": True},
                {"id": "two", "choice_id": "yes", "probabilities": {"yes": 0.6, "no": 0.4},
                 "requires_review": False},
            ]}))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"name": "test", "version": "0.1",
                                            "data_sha256": digest(data.read_bytes())}))
            result = benchmark(SimpleNamespace(
                data=str(data), predictions=str(predictions), output=str(output),
                manifest=str(manifest), split="test", thresholds=[0.5, 0.9]))
            self.assertEqual(result["rows"], 2)
            report = json.loads(output.read_text())
            self.assertEqual(report["metrics"]["selected_correct"], 1)
            self.assertAlmostEqual(report["metrics"]["review_rate"], 0.5)
            self.assertEqual(report["metrics"]["selective"]["0.9"]["selected_count"], 0)
            self.assertIn("knowledge", report["by_family"])
            self.assertIn("target_cross_entropy", report["metrics"])

    def test_benchmark_rejects_missing_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data.jsonl"
            predictions = root / "predictions.json"
            rows = [_row("one", "knowledge", "yes")]
            data.write_text(json.dumps(rows[0]) + "\n")
            predictions.write_text(json.dumps({"predictions": []}))
            with self.assertRaises(ValueError):
                benchmark(SimpleNamespace(
                    data=str(data), predictions=str(predictions),
                    output=str(root / "report.json"), manifest=None, split=None, thresholds=None))


if __name__ == "__main__":
    unittest.main()
