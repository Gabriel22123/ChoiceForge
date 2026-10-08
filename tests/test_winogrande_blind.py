import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from prepare_winogrande_blind import select_rows, selection_key


class WinoGrandeBlindTests(unittest.TestCase):
    def records(self):
        return [{"qID": f"q{i}", "sentence": f"A{i} thanked _ for helping.",
                 "option1": f"person {i}a", "option2": f"person {i}b",
                 "answer": "1" if i % 2 else "2", "_official_index": i}
                for i in range(12)]

    def test_selection_does_not_read_answers(self):
        before = self.records()
        after = copy.deepcopy(before)
        for row in after:
            row["answer"] = "2" if row["answer"] == "1" else "1"
        self.assertEqual([selection_key(row) for row in before], [selection_key(row) for row in after])
        self.assertEqual([row["id"] for row in select_rows(before, 6)],
                         [row["id"] for row in select_rows(after, 6)])

    def test_rows_keep_labels_outside_model_input(self):
        rows = select_rows(self.records(), 8)
        self.assertTrue(all(row["label"] in ("option1", "option2") for row in rows))
        self.assertTrue(all(set(row["request"]) == {"task", "context", "choices"} for row in rows))
        self.assertTrue(all(row["evaluation_regime"] == "untouched_task_family" for row in rows))
        with self.assertRaises(ValueError):
            select_rows(self.records(), 0)


if __name__ == "__main__":
    unittest.main()
