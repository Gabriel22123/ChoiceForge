import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_gsm8k_capacity", ROOT / "scripts/prepare_gsm8k_capacity_data.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def row(index):
    return {"question": f"Question {index}?", "answer": f"Work.\n#### {index + 10}"}


class GSM8KCapacityPreparationTest(unittest.TestCase):
    def test_consumed_validation_is_excluded(self):
        source = [row(index) for index in range(20)]
        train, old, validation, reserve = MODULE.capacity_split(source, 4, 3, 10, 2)
        sets = [{key for key, _ in values} for values in (train, old, validation, reserve)]
        self.assertEqual([len(values) for values in (train, old, validation, reserve)],
                         [10, 3, 2, 5])
        for index, left in enumerate(sets):
            for right in sets[index + 1:]:
                self.assertFalse(left & right)

    def test_invalid_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.capacity_split([row(index) for index in range(4)], 2, 1, 1, 1)

    def test_unicode_line_separator_is_normalized_after_answer_blind_selection(self):
        raw = {"question": "First\x85second?", "answer": "Work.\n#### 12"}
        original = MODULE.BASE.selection_key(raw)
        case = MODULE.make_case(raw, "validation", "a" * 40, "b" * 64)
        self.assertNotIn("\x85", case["request"]["context"])
        self.assertEqual(case["provenance"][0]["selection_key"], original)
        self.assertEqual(case["group_id"], "gsm8k:" + original)


if __name__ == "__main__":
    unittest.main()
