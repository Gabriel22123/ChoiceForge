import math
import unittest

import torch

from scripts.task_gradient_control import combine, median_upper_cap, symmetric_pairwise_projection


class TaskGradientControlTest(unittest.TestCase):
    def test_raw_weighted_sum(self):
        gradients = {"a": [torch.tensor([1.0, 2.0])], "b": [torch.tensor([3.0, -2.0])]}
        value, record = combine(gradients, {"a": .25, "b": .75}, "raw")
        torch.testing.assert_close(value[0], torch.tensor([2.5, -1.0]))
        self.assertEqual(record["transformation"]["kind"], "identity")

    def test_median_cap_never_amplifies_small_gradient(self):
        gradients = {
            "small": [torch.tensor([1.0, 0.0])],
            "middle": [torch.tensor([0.0, 2.0])],
            "large": [torch.tensor([30.0, 40.0])],
        }
        transformed, record = median_upper_cap(gradients)
        self.assertEqual(record["target_norm"], 2.0)
        self.assertEqual(record["scales"]["small"], 1.0)
        self.assertEqual(record["scales"]["middle"], 1.0)
        self.assertAlmostEqual(record["scales"]["large"], .04)
        self.assertAlmostEqual(torch.linalg.vector_norm(transformed["large"][0]).item(), 2.0, places=6)
        self.assertEqual(record["amplified_tasks"], [])

    def test_symmetric_projection_removes_two_task_conflict(self):
        gradients = {"a": [torch.tensor([1.0, 0.0])], "b": [torch.tensor([-1.0, 1.0])]}
        transformed, record = symmetric_pairwise_projection(gradients)
        self.assertEqual(record["conflicting_pairs"], ["a:b"])
        self.assertAlmostEqual(float(torch.dot(transformed["a"][0], gradients["b"][0])), 0.0)
        self.assertAlmostEqual(float(torch.dot(transformed["b"][0], gradients["a"][0])), 0.0)

    def test_projection_is_identity_without_conflict(self):
        gradients = {
            "a": [torch.tensor([1.0, 0.0])],
            "b": [torch.tensor([1.0, 1.0])],
            "c": [torch.tensor([0.0, 2.0])],
        }
        transformed, record = symmetric_pairwise_projection(gradients)
        self.assertEqual(record["conflicting_pairs"], [])
        for name in gradients:
            torch.testing.assert_close(transformed[name][0], gradients[name][0])

    def test_invalid_weights_are_rejected(self):
        gradients = {"a": [torch.tensor([1.0])], "b": [torch.tensor([2.0])]}
        for weights in ({"a": .5}, {"a": .6, "b": .6}, {"a": -1.0, "b": 2.0}):
            with self.assertRaises(ValueError):
                combine(gradients, weights, "raw")


if __name__ == "__main__":
    unittest.main()
