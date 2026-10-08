import unittest

import torch

from scripts.task_gradient_trust_projection import combine, trust_blend


class TrustProjectionTest(unittest.TestCase):
    def test_trust_blend_uses_largest_boundary_intersection(self):
        reference = {"a": [torch.tensor([1.0, 0.0])], "b": [torch.tensor([1.0, 0.0])]}
        projected = {"a": [torch.tensor([-1.0, 1.0])], "b": [torch.tensor([-1.0, 1.0])]}
        weights = {"a": 0.5, "b": 0.5}
        _, combined, record = trust_blend(reference, projected, weights)
        self.assertAlmostEqual(record["alpha"], 0.8 * (1 - 1e-7), places=6)
        self.assertTrue(record["active"])
        self.assertLessEqual(torch.linalg.vector_norm(combined[0]).item(), 1.0 + 1e-6)

    def test_trust_blend_keeps_full_projection_when_norm_does_not_rise(self):
        reference = {"a": [torch.tensor([1.0, 0.0])], "b": [torch.tensor([1.0, 0.0])]}
        projected = {"a": [torch.tensor([0.0, 0.5])], "b": [torch.tensor([0.0, 0.5])]}
        weights = {"a": 0.5, "b": 0.5}
        _, combined, record = trust_blend(reference, projected, weights)
        self.assertEqual(record["alpha"], 1.0)
        self.assertFalse(record["active"])
        self.assertTrue(torch.equal(combined[0], torch.tensor([0.0, 0.5])))

    def test_end_to_end_bound_and_raw_identity(self):
        gradients = {
            "paws": [torch.tensor([4.0, -1.0, 2.0])],
            "snli": [torch.tensor([-3.0, 2.0, 1.0])],
            "replay": [torch.tensor([0.2, -0.1, 0.0])],
        }
        weights = {"paws": 3 / 8, "snli": 3 / 8, "replay": 2 / 8}
        _, record = combine(gradients, weights, "soft_cap_2x_project_trust")
        trust = record["transformation"]["trust_region"]
        self.assertLessEqual(record["combined_norm"], trust["bound_norm"] + 1e-6)
        raw, raw_record = combine(gradients, weights, "raw")
        expected = sum(gradients[name][0] * weights[name] for name in gradients)
        self.assertTrue(torch.equal(raw[0], expected))
        self.assertEqual(raw_record["transformation"]["kind"], "identity")


if __name__ == "__main__":
    unittest.main()
