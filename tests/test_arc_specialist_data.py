import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/prepare_arc_specialist_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_arc_specialist_data", SCRIPT)
ARC = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ARC)


def raw(identity, answer="A", question="What changes?"):
    return {"id": identity, "question": question,
            "choices": {"label": ["A", "B", "C", "D"],
                        "text": ["alpha", "beta", "gamma", "delta"]},
            "answerKey": answer}


class ArcSpecialistDataTests(unittest.TestCase):

    def test_selection_and_candidate_order_are_answer_blind(self):
        first, second = raw("x", "A"), raw("x", "C")
        self.assertEqual(ARC.selection_key(first, "arc-easy"),
                         ARC.selection_key(second, "arc-easy"))
        first_request, first_label = ARC.request_and_label(first)
        second_request, second_label = ARC.request_and_label(second)
        self.assertEqual(first_request, second_request)
        self.assertNotEqual(first_label, second_label)
        self.assertFalse(any(choice["id"] in {"A", "B", "C", "D"}
                             for choice in first_request["choices"]))

    def test_selection_excludes_validation_and_cross_source_duplicates(self):
        keep_a = raw("a", question="unique a")
        keep_b = raw("b", question="unique b")
        blocked = raw("blocked", question="blocked")
        duplicate_a = raw("dup-a", question="duplicate")
        duplicate_b = raw("dup-b", question="duplicate")
        blocked_key = ARC.semantic_key(ARC.request_and_label(blocked)[0])
        selected, summary = ARC.select_rows(
            {"arc-challenge": [keep_a, blocked, duplicate_a],
             "arc-easy": [keep_b, duplicate_b]},
            1, {blocked_key}, lambda request: True)
        self.assertEqual({row["id"] for _, row in selected}, {"a", "b"})
        self.assertEqual(summary["arc-challenge"]["selected_rows"], 1)
        self.assertEqual(summary["arc-easy"]["selected_rows"], 1)

    def test_case_has_public_provenance_and_no_source_answer_letters(self):
        value = ARC.case(raw("x", "B"), "arc-easy", "revision", "a" * 64)
        self.assertEqual(value["split"], "train")
        self.assertEqual(value["family"], "grade_school_science_choice")
        self.assertEqual(value["provenance"][0]["license"], "CC-BY-SA-4.0")
        self.assertIn(value["label"], {choice["id"] for choice in value["request"]["choices"]})


if __name__ == "__main__":
    unittest.main()
