import unittest

from decision_model.train import evaluation_splits


class TrainingProbeTests(unittest.TestCase):
    def test_probe_never_changes_training_or_held_out_selection(self):
        rows = [{"id": str(i), "source": "s", "family": "f", "label": str(i % 2)} for i in range(10)]
        splits = {"train": rows, "validation": rows[:3], "test": rows[3:5]}
        selected = evaluation_splits(splits, {"max_train_evaluation_rows": 4})
        self.assertEqual(len(selected["train"]), 4)
        self.assertEqual(len(splits["train"]), 10)
        self.assertIs(selected["validation"], splits["validation"])
        self.assertIs(selected["test"], splits["test"])
        self.assertEqual({r["label"] for r in selected["train"]}, {"0", "1"})
        self.assertEqual(len(evaluation_splits(splits, {})["train"]), 10)
        self.assertIs(evaluation_splits(splits, {})["train"], splits["train"])

    def test_invalid_probe_caps_fail(self):
        for cap in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                evaluation_splits({"train": [], "validation": [], "test": []}, {"max_train_evaluation_rows": cap})
