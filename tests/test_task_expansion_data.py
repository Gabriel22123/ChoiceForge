import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from task_expansion_data import (boolq_case, boolq_request, boolq_selection_key,
                                 select_balanced_paws, select_boolq)


class TaskExpansionDataTests(unittest.TestCase):
    def test_boolq_request_does_not_read_answer(self):
        request = boolq_request({"question": "Is this supported?", "passage": "Evidence."})
        self.assertEqual([choice["id"] for choice in request["choices"]], ["no", "yes"])

    def test_boolq_selection_never_reads_answer(self):
        rows = [{"question": f"question {i}", "passage": f"passage {i}", "answer": bool(i % 2)}
                for i in range(20)]
        changed = copy.deepcopy(rows)
        for row in changed:
            row["answer"] = not row["answer"]
        self.assertEqual([boolq_selection_key(row) for row in rows],
                         [boolq_selection_key(row) for row in changed])
        self.assertEqual([boolq_selection_key(row) for row in select_boolq(rows, 8)],
                         [boolq_selection_key(row) for row in select_boolq(changed, 8)])

    def test_gold_label_is_outside_boolq_model_input(self):
        row = {"question": "is the claim true", "passage": "The claim is true.", "answer": True}
        case = boolq_case(row, "x")
        self.assertEqual(set(case["request"]), {"task", "context", "choices"})
        self.assertNotIn("answer", case["request"]["context"])
        self.assertEqual(case["label"], "yes")

    def test_paws_selection_is_balanced_and_distinct(self):
        rows = [{"id": i, "sentence1": f"a {i}", "sentence2": f"b {i}", "label": i % 2}
                for i in range(20)]
        selected = select_balanced_paws(rows, 5)
        self.assertEqual(sum(row["label"] == 0 for row in selected), 5)
        self.assertEqual(sum(row["label"] == 1 for row in selected), 5)
        self.assertEqual(len({row["id"] for row in selected}), 10)


if __name__ == "__main__":
    unittest.main()
