import unittest

import torch

from decision_model.counterfactual_routing import composite_gate_costs, optimal_gate


class CounterfactualRoutingTests(unittest.TestCase):

    def test_behavioral_target_opens_helpful_expert_and_closes_harmful_expert(self):
        gates = torch.linspace(0, 1, 21)
        parent = torch.tensor([0., 0.])
        null = torch.tensor([0., 0.])
        helpful = composite_gate_costs(
            parent, null, torch.tensor([-2., 2.]), torch.zeros(2), 1, gates,
            context_weight=0.)
        harmful = composite_gate_costs(
            parent, null, torch.tensor([2., -2.]), torch.zeros(2), 1, gates,
            context_weight=0.)
        self.assertEqual(optimal_gate(helpful, gates)["gate"], 1.0)
        self.assertEqual(optimal_gate(harmful, gates)["gate"], 0.0)

    def test_teacher_cost_can_keep_a_gold_improving_expert_closed(self):
        gates = torch.tensor([0., 1.])
        parent = torch.tensor([0., 0.])
        correction = torch.tensor([-0.2, 0.2])
        teacher = torch.tensor([0.5, 0.5])
        costs = composite_gate_costs(
            parent, parent, correction, torch.zeros(2), 1, gates,
            brier_weight=0., context_weight=0., teacher=teacher,
            distillation_weight=20.)
        self.assertEqual(optimal_gate(costs, gates)["gate"], 0.0)


if __name__ == "__main__":
    unittest.main()
