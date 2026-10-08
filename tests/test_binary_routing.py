import unittest

import torch

from decision_model.binary_routing import binary_route_target, hard_route


class BinaryRoutingTests(unittest.TestCase):
    def test_target_opens_only_when_expert_cost_is_strictly_lower(self):
        gates = torch.tensor([0.0, 1.0])
        self.assertEqual(binary_route_target(torch.tensor([2.0, 1.0]), gates)["gate"], 1)
        self.assertEqual(binary_route_target(torch.tensor([1.0, 1.0]), gates)["gate"], 0)
        self.assertEqual(binary_route_target(torch.tensor([1.0, 2.0]), gates)["gate"], 0)

    def test_hard_route_uses_frozen_half_probability_boundary(self):
        values = hard_route(torch.tensor([0.0, 0.4999, 0.5, 1.0]))
        self.assertTrue(torch.equal(values, torch.tensor([0.0, 0.0, 1.0, 1.0])))

    def test_invalid_binary_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            binary_route_target(torch.tensor([1.0, 0.0, 2.0]), torch.tensor([0.0, 0.5, 1.0]))
        with self.assertRaises(ValueError):
            hard_route(torch.tensor([1.1]))


if __name__ == "__main__":
    unittest.main()
