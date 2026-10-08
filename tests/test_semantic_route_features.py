import unittest

import torch

from decision_model.semantic_route_features import fixed_projection, semantic_route_features


class SemanticRouteFeatureTests(unittest.TestCase):

    def test_projection_is_deterministic_and_has_expected_scale(self):
        first = fixed_projection(8, 4, 1701)
        second = fixed_projection(8, 4, 1701)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (8, 4))

    def test_features_are_candidate_order_invariant(self):
        generator = torch.Generator().manual_seed(5)
        full = torch.randn(3, 8, generator=generator)
        null = torch.randn(3, 8, generator=generator)
        projection = fixed_projection(8, 4, 1701)
        baseline = semantic_route_features(full, null, projection)
        reordered = semantic_route_features(full[[2, 0, 1]], null[[2, 0, 1]], projection)
        self.assertTrue(torch.allclose(baseline, reordered, atol=1e-6, rtol=0))
        self.assertEqual(len(baseline), 16)

    def test_invalid_candidate_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            semantic_route_features(torch.zeros(2, 8), torch.zeros(3, 8),
                                    fixed_projection(8, 4))


if __name__ == "__main__": unittest.main()
