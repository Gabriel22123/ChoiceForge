import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_gsm8k_specialist", ROOT / "scripts/prepare_gsm8k_specialist_data.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class GSM8KPreparationTest(unittest.TestCase):
    def setUp(self):
        self.row = {
            "question": "Mina has 48 apples and buys 24 more. How many apples now?",
            "answer": "She buys 48/2 = <<48/2=24>>24.\nThen 48+24 = <<48+24=72>>72.\n#### 72",
        }

    def test_split_key_ignores_answer(self):
        changed = dict(self.row, answer="Different work.\n#### 99")
        self.assertEqual(MODULE.selection_key(self.row), MODULE.selection_key(changed))

    def test_rationale_is_excluded_and_correct_choice_is_preserved(self):
        request, label = MODULE.request_and_label(self.row)
        self.assertNotIn("She buys", request["context"])
        self.assertEqual(len(request["choices"]), 4)
        self.assertEqual(len({choice["description"] for choice in request["choices"]}), 4)
        answer = next(choice["description"] for choice in request["choices"]
                      if choice["id"] == label)
        self.assertEqual(answer, "72")

    def test_decimal_normalization(self):
        row = dict(self.row, answer="Work.\n#### 1,200.50")
        correct, distractors = MODULE.numeric_candidates(row)
        self.assertEqual(correct, "1200.5")
        self.assertEqual(len(distractors), 3)
        self.assertNotIn(correct, distractors)


if __name__ == "__main__":
    unittest.main()
