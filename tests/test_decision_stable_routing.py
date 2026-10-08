import unittest

import torch

from decision_model.decision_stable_routing import add_parent_choice_cost


class DecisionStableRoutingTests(unittest.TestCase):

    def test_penalizes_each_changed_parent_choice_only_when_active(self):
        gates = torch.tensor([0., 0.5, 1.])
        parent = torch.tensor([1., 0.])
        null = torch.tensor([0., 1.])
        correction = torch.tensor([-2., 2.])
        null_correction = torch.tensor([2., -2.])
        base = torch.zeros(3)
        actual = add_parent_choice_cost(
            base, parent, null, correction, null_correction, gates,
            active=True, change_cost=1.)
        self.assertEqual(actual.tolist(), [0., 2., 2.])
        inactive = add_parent_choice_cost(
            base, parent, null, correction, null_correction, gates,
            active=False, change_cost=1.)
        self.assertTrue(torch.equal(inactive, base))


if __name__ == "__main__":
    unittest.main()
