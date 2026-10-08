import importlib.util
import json
import unittest
from pathlib import Path

from decision_model.core import digest


SCRIPT = Path(__file__).parents[1] / "scripts/run_extended_semantic_consensus_route.py"
SPEC = importlib.util.spec_from_file_location("run_extended_semantic_consensus_route", SCRIPT)
ROUTE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ROUTE)


class ExtendedSemanticConsensusRouteTests(unittest.TestCase):
    def test_protocol_freezes_seven_sources_five_experts_and_original_gates(self):
        root = Path(__file__).parents[1]
        extended = json.loads((root / "configs/extended-semantic-consensus-route-v1.json").read_text())
        original = json.loads((root / "configs/semantic-consensus-route-lofo-v1.json").read_text())
        self.assertEqual(extended["status"], "frozen_before_training")
        self.assertEqual(sum(len(group["sources"]) for group in extended["groups"]), 7)
        self.assertEqual(len(extended["experts"]), 5)
        for key in ("violation_rate_max", "safe_selection_precision_min",
                    "expert_selection_rate_min", "proper_gain_min"):
            self.assertEqual(extended["screening_rule"][key], original["screening_rule"][key])
        for path, expected in extended["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_group_loader_and_consensus_evaluator_are_available(self):
        self.assertTrue(callable(ROUTE.load_route_group))
        self.assertTrue(callable(ROUTE.evaluate_consensus))


if __name__ == "__main__": unittest.main()
