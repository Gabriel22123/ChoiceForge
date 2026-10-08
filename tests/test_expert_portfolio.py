import unittest

import torch

from decision_model.expert_portfolio import (
    decision_stable_projection, source_robust_objective, validate_alpha_grid)


class ExpertPortfolioTest(unittest.TestCase):
    def test_projection_uses_largest_stable_alpha_and_shares_it(self):
        parent = torch.tensor([2.0, 1.0])
        parent_null = torch.tensor([1.5, 1.0])
        expert = torch.tensor([0.0, 3.0])
        expert_null = torch.tensor([0.5, 2.5])
        full, null, alpha = decision_stable_projection(
            parent, parent_null, expert, expert_null, (0.25, 0.5, 0.75, 1.0))
        self.assertEqual(0.25, alpha)
        self.assertEqual(0, int(full.argmax()))
        self.assertTrue(torch.equal(null, 0.75 * parent_null + 0.25 * expert_null))

    def test_projection_is_permutation_equivariant(self):
        order = torch.tensor([2, 0, 1])
        values = [torch.tensor([2.0, 0.0, 1.0]), torch.tensor([1.0, 0.5, 0.2]),
                  torch.tensor([1.5, 0.2, 1.4]), torch.tensor([0.9, 0.1, 0.8])]
        original = decision_stable_projection(*values, (0.25, 0.5, 1.0))
        permuted = decision_stable_projection(
            *(value[order] for value in values), (0.25, 0.5, 1.0))
        self.assertEqual(original[2], permuted[2])
        self.assertTrue(torch.equal(original[0][order], permuted[0]))
        self.assertTrue(torch.equal(original[1][order], permuted[1]))

    def test_source_objective_penalizes_proper_and_context_regression(self):
        loss, details = source_robust_objective(
            torch.tensor([1.1, 0.8]), torch.tensor([1.0, 1.0]),
            torch.tensor([0.2, 0.5]), torch.tensor([0.3, 0.4]), 10.0, 5.0)
        self.assertGreater(float(loss), 0.95)
        self.assertAlmostEqual(0.1, float(details["proper_regression_max"]), places=6)
        self.assertAlmostEqual(0.1, float(details["context_regression_max"]), places=6)

    def test_alpha_grid_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            validate_alpha_grid((0.5, 0.5, 1.0))


if __name__ == "__main__":
    unittest.main()
