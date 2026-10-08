import unittest

import torch

from scripts.task_gradient_composition import combine, soft_median_cap


class TaskGradientCompositionTest(unittest.TestCase):
    def setUp(self):
        self.gradients = {
            "a": [torch.tensor([1.0, 0.0])],
            "b": [torch.tensor([0.0, 2.0])],
            "c": [torch.tensor([-8.0, 0.0])],
        }
        self.weights = {"a": 0.25, "b": 0.25, "c": 0.5}

    def test_soft_cap_uses_twice_median_without_amplification(self):
        transformed, record = soft_median_cap(self.gradients)
        self.assertEqual(record["median_norm"], 2.0)
        self.assertEqual(record["threshold_norm"], 4.0)
        self.assertEqual(record["scales"], {"a": 1.0, "b": 1.0, "c": 0.5})
        self.assertTrue(torch.equal(transformed["a"][0], self.gradients["a"][0]))
        self.assertTrue(torch.equal(transformed["c"][0], torch.tensor([-4.0, 0.0])))

    def test_composition_caps_before_projecting(self):
        _, record = combine(self.gradients, self.weights, "soft_cap_2x_then_project")
        transform = record["transformation"]
        self.assertEqual(transform["kind"], "two_x_median_upper_cap_then_symmetric_projection")
        self.assertEqual(transform["cap"]["scales"]["c"], 0.5)
        self.assertIn("a:c", transform["projection"]["conflicting_pairs"])
        self.assertGreaterEqual(record["after"]["cosines"]["a:c"], -1e-7)

    def test_raw_is_identity(self):
        combined, record = combine(self.gradients, self.weights, "raw")
        expected = sum(self.gradients[name][0] * self.weights[name] for name in self.gradients)
        self.assertTrue(torch.equal(combined[0], expected))
        self.assertEqual(record["transformation"]["kind"], "identity")

    def test_rejects_too_small_multiplier_and_unknown_method(self):
        with self.assertRaises(ValueError):
            soft_median_cap(self.gradients, 0.5)
        with self.assertRaises(ValueError):
            combine(self.gradients, self.weights, "unknown")


if __name__ == "__main__":
    unittest.main()
