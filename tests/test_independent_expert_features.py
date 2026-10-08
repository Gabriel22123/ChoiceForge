import unittest

import torch

from decision_model.modular_route_features import independent_expert_features


class IndependentExpertFeatureTests(unittest.TestCase):
    def test_other_expert_blocks_cannot_change_selected_module_input(self):
        names = ("a", "b", "c")
        combined = torch.arange(7 + 3 * 17 + 8, dtype=torch.float32)
        baseline = independent_expert_features(combined, names, "b", 8)
        changed = combined.clone(); changed[7:24] += 1000; changed[41:58] -= 1000
        self.assertTrue(torch.equal(baseline, independent_expert_features(changed, names, "b", 8)))
        self.assertEqual(len(baseline), 32)

    def test_width_and_expert_are_validated(self):
        with self.assertRaises(ValueError):
            independent_expert_features(torch.zeros(10), ("a",), "a", 0)
        with self.assertRaises(ValueError):
            independent_expert_features(torch.zeros(24), ("a",), "missing", 0)


if __name__ == "__main__": unittest.main()
