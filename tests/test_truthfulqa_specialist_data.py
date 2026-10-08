import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_truthfulqa_specialist_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_truthfulqa_specialist_data", SCRIPT)
TQA = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(TQA)


def raw(question, correct="truth", incorrect="falsehood"):
    return {"Type": "Adversarial", "Category": "Misconceptions", "Question": question,
            "Best Answer": correct, "Best Incorrect Answer": incorrect,
            "Correct Answers": correct, "Incorrect Answers": incorrect, "Source": "fixture"}


class TruthfulQaSpecialistDataTests(unittest.TestCase):

    def test_split_and_order_are_answer_blind(self):
        first = raw("q", "truth", "falsehood")
        swapped = raw("q", "falsehood", "truth")
        self.assertEqual(TQA.selection_key(first), TQA.selection_key(swapped))
        first_request, first_label = TQA.request_and_label(first)
        second_request, second_label = TQA.request_and_label(swapped)
        self.assertEqual(first_request, second_request)
        self.assertNotEqual(first_label, second_label)

    def test_split_is_deterministic_and_keeps_reserve(self):
        rows = [raw(f"question {index}") for index in range(8)]
        first = TQA.split_rows(rows, 4, 3)
        second = TQA.split_rows(list(reversed(rows)), 4, 3)
        self.assertEqual([[key for key, _ in part] for part in first],
                         [[key for key, _ in part] for part in second])
        self.assertEqual([len(part) for part in first], [4, 3, 1])

    def test_case_has_binary_contract_and_public_provenance(self):
        value = TQA.make_case(raw("q"), "train", "r" * 40, "a" * 64)
        self.assertEqual(value["decision_type"], "boolean")
        self.assertEqual(len(value["request"]["choices"]), 2)
        self.assertEqual(value["provenance"][0]["license"], "Apache-2.0")
        self.assertIn(value["label"], {choice["id"] for choice in value["request"]["choices"]})


if __name__ == "__main__":
    unittest.main()
