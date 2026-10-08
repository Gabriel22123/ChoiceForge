import importlib.util
import json
import unittest
from pathlib import Path

import torch

from decision_model.core import digest


SCRIPT = Path(__file__).parents[1] / "scripts/run_factored_safe_route_lofo.py"
SPEC = importlib.util.spec_from_file_location("run_factored_safe_route_lofo", SCRIPT)
ROUTE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ROUTE)


class FactoredSafeRouteTests(unittest.TestCase):

    def test_protocol_is_frozen_with_nested_calibration(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "configs/factored-safe-route-lofo-v1.json").read_text())
        self.assertEqual(config["status"], "frozen_before_training")
        self.assertLess(config["training"]["calibration_updates"],
                        config["training"]["final_updates"])
        self.assertGreater(config["screening_rule"]["expert_selection_rate_min"], 0)
        for path, expected in config["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_safety_targets_are_independent_per_expert(self):
        rows = [{"id": "one"}]
        parent = {"correct": True, "cross_entropy": 1.0, "brier": 0.4,
                  "context_advantage": 0.1, "proper_cost": 1.2}
        safe = {"correct": True, "cross_entropy": 0.8, "brier": 0.3,
                "context_advantage": 0.2, "proper_cost": 0.95}
        unsafe = {"correct": False, "cross_entropy": 0.7, "brier": 0.2,
                  "context_advantage": 0.3, "proper_cost": 0.8}
        labels, gains = ROUTE.safety_and_gain_targets(
            rows, {"one": {"parent": parent, "experts": {"a": safe, "b": unsafe}}},
            ("a", "b"))
        self.assertEqual(labels.tolist(), [[1.0, 0.0]])
        self.assertAlmostEqual(float(gains[0, 0]), 0.25, places=6)
        self.assertAlmostEqual(float(gains[0, 1]), 0.4, places=6)

    def test_positive_weights_are_finite_and_capped(self):
        labels = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        weights = ROUTE.positive_weights(labels, 2.0)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())
        self.assertLessEqual(float(weights.max()), 2.0)

    def test_factored_router_has_separate_safety_and_gain_outputs(self):
        module = ROUTE.FactoredRouter(5, 7, 3)
        safety, gain = module(torch.zeros(2, 5))
        self.assertEqual(tuple(safety.shape), (2, 3))
        self.assertEqual(tuple(gain.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
