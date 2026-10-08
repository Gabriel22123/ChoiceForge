import unittest

import torch

from decision_model.behavioral_route_features import (
    FEATURE_NAMES, behavioral_route_features, standardize_behavior_features)


class BehavioralRouteFeatureTests(unittest.TestCase):
    def setUp(self):
        self.parent = torch.tensor([1.0, 0.3, -0.2])
        self.parent_null = torch.tensor([0.2, 0.1, 0.0])
        self.correction = torch.tensor([-0.8, 0.9, 0.2])
        self.null_correction = torch.tensor([0.1, -0.2, 0.3])

    def test_features_are_candidate_permutation_invariant(self):
        original = behavioral_route_features(
            self.parent, self.parent_null, self.correction, self.null_correction)
        order = torch.tensor([2, 0, 1])
        permuted = behavioral_route_features(
            self.parent[order], self.parent_null[order], self.correction[order],
            self.null_correction[order])
        self.assertEqual(len(original), len(FEATURE_NAMES))
        self.assertTrue(torch.allclose(original, permuted, atol=1e-7, rtol=0))

    def test_features_ignore_additive_residual_constants(self):
        original = behavioral_route_features(
            self.parent, self.parent_null, self.correction, self.null_correction)
        shifted = behavioral_route_features(
            self.parent, self.parent_null, self.correction + 7, self.null_correction - 4)
        self.assertTrue(torch.allclose(original, shifted, atol=1e-6, rtol=0))

    def test_standardization_fits_only_provided_training_matrix(self):
        training = torch.stack([
            behavioral_route_features(
                self.parent, self.parent_null, self.correction, self.null_correction),
            behavioral_route_features(
                self.parent * 2, self.parent_null, self.correction, self.null_correction),
        ])
        transformed, mean, scale = standardize_behavior_features(
            training, [training, training[:1]])
        self.assertTrue(torch.allclose(transformed[0].mean(0), torch.zeros(len(FEATURE_NAMES)),
                                       atol=1e-5, rtol=0))
        self.assertEqual(mean.shape, scale.shape)
        self.assertEqual(transformed[1].shape, (1, len(FEATURE_NAMES)))

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            behavioral_route_features(
                self.parent, self.parent_null[:2], self.correction, self.null_correction)


if __name__ == "__main__":
    unittest.main()
