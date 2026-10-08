import importlib.util
import unittest
from pathlib import Path


def load_script():
    path = Path(__file__).parents[1] / "scripts/prepare_mbpp_outcome_screen.py"
    spec = importlib.util.spec_from_file_location("prepare_outcome_screen", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(range(len(text.split()) + int(add_special_tokens)))


class PrepareOutcomeScreenTests(unittest.TestCase):
    def test_request_level_gate_can_exclude_many_short_individual_paths(self):
        module = load_script()
        def row(row_id, choices):
            return {"id": row_id, "request": {"task": "pick", "context": "case",
                                                "choices": [{"id": str(i), "description": value}
                                                            for i, value in enumerate(choices)]}}
        short = row("short", ["a", "b"])
        aggregate = row("aggregate", [f"one two three {index}" for index in range(5)])
        kept, excluded, admissions, paths = module.admitted(
            [short, aggregate], FakeTokenizer(), max_tokens=10)
        self.assertEqual([item["id"] for item in kept], ["short"])
        self.assertEqual(excluded["request_admission_over_limit"], 1)
        self.assertLess(max(paths), max(admissions))


if __name__ == "__main__":
    unittest.main()
