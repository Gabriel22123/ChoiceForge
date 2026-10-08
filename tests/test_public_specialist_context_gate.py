import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "public_specialist_context_gate", ROOT / "scripts/run_public_specialist_context_gate.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def seed(value, parent=0.1, expert=0.2):
    return {"seed": value, "evaluation": {
        "context_advantage": {"source": {"mean": expert}},
        "parent_context_advantage": {"source": {"mean": parent}},
    }}


class PublicSpecialistContextGateTest(unittest.TestCase):
    def test_all_seeds_must_pass(self):
        passed, details = MODULE.hard_context_gate([seed(42), seed(43, expert=0.09)], 0.0)
        self.assertFalse(passed)
        self.assertEqual([item["seed"] for item in details], [42, 43])

    def test_source_coverage_must_match(self):
        item = seed(42)
        item["evaluation"]["parent_context_advantage"] = {"other": {"mean": 0.1}}
        with self.assertRaises(ValueError):
            MODULE.hard_context_gate([item], 0.0)


if __name__ == "__main__":
    unittest.main()
