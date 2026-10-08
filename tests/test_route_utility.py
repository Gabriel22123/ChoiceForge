import math
import unittest

import torch

from decision_model.route_utility import signed_route_utility, utility_route_metrics


class RouteUtilityTests(unittest.TestCase):
    def test_signed_utility_uses_zero_as_parent_expert_boundary(self):
        gates = torch.tensor([0.0, 1.0])
        open_target = signed_route_utility(torch.tensor([2.0, 1.0]), gates)
        closed_target = signed_route_utility(torch.tensor([1.0, 2.0]), gates)
        tie = signed_route_utility(torch.tensor([1.0, 1.0]), gates)
        self.assertEqual((open_target["gate"], closed_target["gate"], tie["gate"]), (1, 0, 0))
        self.assertAlmostEqual(open_target["transformed_utility"], math.asinh(1.0))

    def test_route_regret_weights_wrong_sign_by_counterfactual_cost(self):
        metrics = utility_route_metrics([2.0, -1.0, 0.5], [1.0, 0.2, -0.1])
        self.assertAlmostEqual(metrics["mean_regret"], 0.5)
        self.assertAlmostEqual(metrics["always_parent_regret"], 2.5 / 3)
        self.assertAlmostEqual(metrics["always_expert_regret"], 1.0 / 3)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            signed_route_utility(torch.tensor([1.0]), torch.tensor([0.0]))
        with self.assertRaises(ValueError):
            utility_route_metrics([1.0], [float("nan")])


if __name__ == "__main__":
    unittest.main()
