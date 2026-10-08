import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_hellaswag_specialist", ROOT / "scripts/prepare_hellaswag_specialist_data.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def raw(label="2"):
    return {
        "ind": 7, "activity_label": "Making tea", "ctx_a": "A person boils water.",
        "ctx_b": "then", "ctx": "A person boils water. then",
        "endings": ["the cup disappears.", "the car starts.",
                    "the person pours the water into a cup.", "the wall sings."],
        "source_id": "wikihow~tea", "split": "train", "split_type": "indomain",
        "label": label,
    }


class HellaSwagPreparationTest(unittest.TestCase):
    def test_selection_ignores_label_and_candidate_order_is_deterministic(self):
        left, right = raw("2"), raw("0")
        self.assertEqual(MODULE.selection_key(left), MODULE.selection_key(right))
        request, label = MODULE.request_and_label(left)
        self.assertEqual(request, MODULE.request_and_label(left)[0])
        answer = next(choice["description"] for choice in request["choices"]
                      if choice["id"] == label)
        self.assertEqual(answer, "the person pours the water into a cup.")

    def test_rejects_malformed_candidates(self):
        value = raw(); value["endings"][1] = value["endings"][0]
        with self.assertRaises(ValueError):
            MODULE.selection_key(value)


if __name__ == "__main__":
    unittest.main()
