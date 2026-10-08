import unittest

import torch

from decision_model.expert_bank_features import (
    expert_bank_feature_names, expert_bank_features, standardize_bank_features)
from decision_model.safe_routing import choose_safe_path, path_statistics


class ExpertBankFeatureTests(unittest.TestCase):

    def test_feature_vector_is_order_invariant_and_has_named_width(self):
        parent = torch.tensor([.2, -.1, .5])
        parent_null = torch.tensor([.0, .1, -.2])
        corrections = {"a": torch.tensor([.1, -.2, .3]),
                       "b": torch.tensor([-.1, .4, .2])}
        null_corrections = {"a": torch.tensor([.2, -.1, .0]),
                            "b": torch.tensor([.3, .1, -.2])}
        value = expert_bank_features(parent, parent_null, corrections, null_corrections)
        order = torch.tensor([2, 0, 1])
        permuted = expert_bank_features(parent[order], parent_null[order],
                                        {key: item[order] for key, item in corrections.items()},
                                        {key: item[order] for key, item in null_corrections.items()})
        self.assertEqual(len(value), len(expert_bank_feature_names(("a", "b"))))
        self.assertTrue(torch.allclose(value, permuted, atol=1e-6))

    def test_standardization_fits_training_only(self):
        train = torch.tensor([[1., 3.], [3., 7.]])
        test = torch.tensor([[5., 11.]])
        values, mean, scale = standardize_bank_features(train, [train, test])
        self.assertTrue(torch.allclose(values[0].mean(0), torch.zeros(2)))
        self.assertEqual(mean.tolist(), [2., 5.])
        self.assertEqual(scale.tolist(), [1., 2.])

    def test_distribution_safe_path_prefers_only_feasible_improvement(self):
        parent = path_statistics([0., 0.], [0., 0.], [.7, .3])
        good = path_statistics([1., 0.], [0., 0.], [.7, .3])
        bad_context = path_statistics([1., 0.], [.84729786, 0.], [.7, .3])
        self.assertEqual(choose_safe_path(parent, {"good": good})[0], "good")
        self.assertEqual(choose_safe_path(parent, {"bad": bad_context}), ("parent", 0.0))


if __name__ == "__main__":
    unittest.main()
