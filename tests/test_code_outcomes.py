import json
import tempfile
import unittest
from pathlib import Path

from decision_model.code_outcomes import (
    UnsafeProgram, deterministic_mutants, evaluate_candidate, load_mbpp, mbpp_split,
    prepare_mbpp_outcomes, run_test, validate_program,
)
from decision_model.core import load_rows, outcome_rewards


class CodeOutcomeTests(unittest.TestCase):
    def test_safe_candidate_passes_and_assertion_is_categorized(self):
        code = "def increment(value):\n    return value + 1"
        outcomes = evaluate_candidate(code, ["assert increment(2) == 3", "assert increment(0) == 0"])
        self.assertEqual([(item.passed, item.category) for item in outcomes], [
            (True, "passed"), (False, "assertion")])

    def test_rejects_dangerous_programs(self):
        for source in (
            "import os\nos.listdir('.')", "open('secret')",
            "getattr(object(), '__class__')", "__import__('math')", "import socket",
        ):
            with self.subTest(source=source), self.assertRaises(UnsafeProgram):
                validate_program(source)

    def test_allows_reviewed_standard_library_imports(self):
        self.assertTrue(validate_program("from math import sqrt\ndef f(x): return sqrt(x)"))

    def test_deterministic_mutants_are_unique_and_one_edit_metadata_is_stable(self):
        code = """def clamp(value):
    if value < 0 or value == 3:
        return value + 1
    return value
"""
        first = deterministic_mutants(code, limit=20)
        second = deterministic_mutants(code, limit=20)
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 5)
        self.assertEqual(len({row["code"] for row in first}), len(first))
        self.assertGreaterEqual({row["operator"] for row in first},
                                {"compare", "binop", "boolop", "integer"})

    def test_rejected_test_never_enters_subprocess(self):
        with self.assertRaises(UnsafeProgram):
            run_test("def f(): return 1", "open('/tmp/x', 'w')")

    def test_load_mbpp_accepts_json_and_jsonl(self):
        row = {"task_id": 601, "text": "increment", "code": "def f(x): return x+1",
               "test_list": ["assert f(1) == 2"]}
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "data.json"
            json_path.write_text("[" + json.dumps(row) + "]")
            jsonl_path = Path(directory) / "data.jsonl"
            jsonl_path.write_text(json.dumps(row) + "\n")
            self.assertEqual(load_mbpp(json_path), [row])
            self.assertEqual(load_mbpp(jsonl_path), [row])

    def test_load_mbpp_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            row = '{"task_id": 601, "text":"x", "code":"pass", "test_list":["assert True"]}'
            path.write_text(row + "\n" + row + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_mbpp(path)

    def test_official_split_boundaries(self):
        self.assertEqual([mbpp_split(value) for value in (1, 10, 11, 510, 511, 600, 601, 974)],
                         ["prompt", "prompt", "test", "test", "validation", "validation", "train", "train"])
        with self.assertRaises(ValueError):
            mbpp_split(975)

    def test_prepare_outcomes_uses_rewards_for_audit_but_not_order(self):
        fixtures = [
            {"task_id": 601, "text": "Return the input plus one.",
             "code": "def increment(value):\n    return value + 1",
             "test_list": ["assert increment(0) == 1", "assert increment(2) == 3"]},
            {"task_id": 511, "text": "Return true for positive values.",
             "code": "def positive(value):\n    return value > 0",
             "test_list": ["assert positive(2)", "assert not positive(0)"]},
            {"task_id": 11, "text": "Excluded test task.",
             "code": "def identity(value):\n    return value",
             "test_list": ["assert identity(2) == 2"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mbpp.json"
            source.write_text(json.dumps(fixtures))
            rows, manifest = prepare_mbpp_outcomes(source, max_candidates=4)
            path = Path(directory) / "data.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(load_rows(path), rows)
        self.assertEqual({row["split"] for row in rows}, {"train", "validation"})
        self.assertEqual(manifest["skipped"], {"split_excluded": 1})
        self.assertFalse(manifest["selection_uses_outcomes"])
        for row in rows:
            ids = [choice["id"] for choice in row["request"]["choices"]]
            self.assertEqual(set(row["candidate_outcomes"]), set(ids))
            rewards = [row["candidate_outcomes"][key]["reward"] for key in ids]
            self.assertEqual(row["label"], ids[max(range(len(ids)), key=rewards.__getitem__)])
            self.assertEqual(outcome_rewards(row), rewards)

    def test_outcome_contract_rejects_inconsistent_counts_and_labels(self):
        row = {
            "request": {"task": "choose", "context": "case", "choices": [
                {"id": "a", "description": "A"}, {"id": "b", "description": "B"}]},
            "label": "a",
            "candidate_outcomes": {
                "a": {"passed": 2, "total": 2, "reward": 1.0,
                      "categories": ["passed", "passed"]},
                "b": {"passed": 1, "total": 2, "reward": 0.5,
                      "categories": ["passed", "assertion"]},
            },
        }
        self.assertEqual(outcome_rewards(row, list(reversed(row["request"]["choices"]))), [.5, 1.])
        broken = json.loads(json.dumps(row))
        broken["candidate_outcomes"]["b"]["reward"] = .75
        with self.assertRaisesRegex(ValueError, "passed/total"):
            outcome_rewards(broken)
        broken = json.loads(json.dumps(row))
        broken["label"] = "b"
        with self.assertRaisesRegex(ValueError, "first maximum"):
            outcome_rewards(broken)


if __name__ == "__main__":
    unittest.main()
