import importlib.util
import json
import unittest
from pathlib import Path

import torch

from decision_model.core import digest


SCRIPT = Path(__file__).parents[1] / "scripts/run_constraint_margin_route_lofo.py"
SPEC = importlib.util.spec_from_file_location("run_constraint_margin_route_lofo", SCRIPT)
ROUTE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ROUTE)


class ConstraintMarginRouteTests(unittest.TestCase):

    def test_protocol_is_frozen_and_uses_conservative_bounds(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "configs/constraint-margin-route-lofo-v1.json").read_text())
        self.assertEqual(config["status"], "frozen_before_training")
        self.assertGreaterEqual(config["training"]["calibration_quantile"], 0.99)
        self.assertGreater(config["screening_rule"]["expert_selection_rate_min"], 0)
        for path, expected in config["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_component_targets_preserve_constraint_signs(self):
        rows = [{"id": "one"}]
        parent = {"correct": True, "cross_entropy": 1.0, "brier": 0.4,
                  "context_advantage": 0.1, "proper_cost": 1.2}
        expert = {"correct": False, "cross_entropy": 0.8, "brier": 0.5,
                  "context_advantage": 0.0, "proper_cost": 1.05}
        risk, margins = ROUTE.component_targets(
            rows, {"one": {"parent": parent, "experts": {"a": expert}}}, ("a",))
        self.assertEqual(risk.tolist(), [[1.0]])
        expected = [-0.2, 0.1, 0.1, 0.15]
        for actual, wanted in zip(margins[0, 0].tolist(), expected):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_upper_quantile_is_conservative_order_statistic(self):
        self.assertEqual(ROUTE.upper_quantile([3, 1, 2, 4], 0.75), 3)
        self.assertEqual(ROUTE.upper_quantile([3, 1, 2, 4], 1.0), 4)

    def test_router_outputs_risk_and_four_margins_per_expert(self):
        module = ROUTE.ConstraintMarginRouter(5, 7, 3)
        risk, margins = module(torch.zeros(2, 5))
        self.assertEqual(tuple(risk.shape), (2, 3))
        self.assertEqual(tuple(margins.shape), (2, 3, 4))


if __name__ == "__main__": unittest.main()
