import importlib.util
import json
import unittest
from pathlib import Path

import torch

from decision_model.core import digest


SCRIPT = Path(__file__).parents[1] / "scripts/run_semantic_consensus_route_lofo.py"
SPEC = importlib.util.spec_from_file_location("run_semantic_consensus_route_lofo", SCRIPT)
ROUTE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ROUTE)


class SemanticConsensusRouteTests(unittest.TestCase):

    def test_protocol_preserves_semantic_v1_gates(self):
        root = Path(__file__).parents[1]
        consensus = json.loads((root / "configs/semantic-consensus-route-lofo-v1.json").read_text())
        single = json.loads((root / "configs/semantic-factored-route-lofo-v1.json").read_text())
        self.assertEqual(consensus["status"], "frozen_before_training")
        for key in ("violation_rate_max", "safe_selection_precision_min",
                    "expert_selection_rate_min", "proper_gain_min"):
            self.assertEqual(consensus["screening_rule"][key], single["screening_rule"][key])
        self.assertEqual(consensus["router_seeds"], [42, 43, 44])
        for path, expected in consensus["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_all_seeds_must_pass_safety_and_positive_gain(self):
        names = ("a", "b")
        predictions = [
            (torch.tensor([[0.9, 0.9]]), torch.tensor([[0.2, 0.4]]), {"a": 0.8, "b": 0.8}),
            (torch.tensor([[0.9, 0.7]]), torch.tensor([[0.1, 0.5]]), {"a": 0.8, "b": 0.8}),
            (torch.tensor([[0.9, 0.9]]), torch.tensor([[0.3, 0.6]]), {"a": 0.8, "b": 0.8}),
        ]
        self.assertEqual(ROUTE.choose_consensus_path(0, names, predictions), "a")
        predictions[2] = (predictions[2][0], torch.tensor([[-0.1, 0.6]]), predictions[2][2])
        self.assertEqual(ROUTE.choose_consensus_path(0, names, predictions), "parent")

    def test_worst_seed_gain_ranks_consensus_experts(self):
        names = ("a", "b")
        thresholds = {"a": 0.5, "b": 0.5}
        predictions = [
            (torch.ones(1, 2), torch.tensor([[0.8, 0.4]]), thresholds),
            (torch.ones(1, 2), torch.tensor([[0.1, 0.3]]), thresholds),
            (torch.ones(1, 2), torch.tensor([[0.7, 0.2]]), thresholds),
        ]
        self.assertEqual(ROUTE.choose_consensus_path(0, names, predictions), "b")


if __name__ == "__main__": unittest.main()
