import unittest

import torch

from decision_model.feature_screen import RoutedResidualHead
from decision_model.merged_adapter import fuse_routed_residual_experts


class MergedAdapterTest(unittest.TestCase):
    def test_fusion_matches_weighted_residuals(self):
        torch.manual_seed(7)
        modules = {
            "wide": RoutedResidualHead.build(5, 7, 3, 0.0, 0.25),
            "small": RoutedResidualHead.build(5, 3, 2, 0.0, 0.25),
        }
        for module in modules.values():
            for parameter in module.parameters():
                parameter.data.normal_()
            module.eval()
        weights = {"wide": 0.3, "small": 0.6}
        merged, metadata = fuse_routed_residual_experts(modules, weights)
        values = torch.randn(4, 5)
        expected = sum(weights[name] * module.residual(values)
                       for name, module in modules.items())
        self.assertTrue(torch.allclose(merged(values), expected, atol=2e-6, rtol=1e-6))
        self.assertEqual(metadata["total_width"], 10)
        self.assertEqual(metadata["expert_order"], ["small", "wide"])

    def test_fusion_rejects_missing_weight(self):
        module = RoutedResidualHead.build(5, 3, 2, 0.0, 0.25)
        with self.assertRaises(ValueError):
            fuse_routed_residual_experts({"one": module}, {})


if __name__ == "__main__":
    unittest.main()
