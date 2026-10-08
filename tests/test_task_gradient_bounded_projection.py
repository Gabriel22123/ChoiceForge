import unittest

import torch

from scripts.task_gradient_bounded_projection import combine, upper_cap_at


class BoundedProjectionTest(unittest.TestCase):
    def test_upper_cap_never_amplifies_and_enforces_threshold(self):
        gradients = {"small": [torch.tensor([0.3, 0.4])],
                     "large": [torch.tensor([3.0, 4.0])]}
        transformed, record = upper_cap_at(gradients, 2.0)
        self.assertEqual(record["scales"], {"small": 1.0, "large": 0.4})
        self.assertTrue(torch.equal(transformed["small"][0], gradients["small"][0]))
        self.assertAlmostEqual(torch.linalg.vector_norm(transformed["large"][0]).item(), 2.0)

    def test_bounded_projection_final_task_norms_do_not_exceed_pre_cap_threshold(self):
        gradients = {
            "paws": [torch.tensor([4.0, -1.0, 2.0])],
            "snli": [torch.tensor([-3.0, 2.0, 1.0])],
            "replay": [torch.tensor([0.2, -0.1, 0.0])],
        }
        weights = {"paws": 3 / 8, "snli": 3 / 8, "replay": 2 / 8}
        _, record = combine(gradients, weights, "soft_cap_2x_project_recap")
        transform = record["transformation"]
        threshold = transform["pre_cap"]["threshold_norm"]
        self.assertEqual(transform["kind"], "two_x_median_cap_project_then_recap")
        self.assertTrue(all(value <= threshold + 1e-6 for value in record["after"]["norms"].values()))
        self.assertTrue(all(0 <= value <= 1 for value in transform["post_cap"]["scales"].values()))

    def test_raw_is_identity_and_unknown_rejected(self):
        gradients = {"a": [torch.tensor([1.0])], "b": [torch.tensor([2.0])]}
        weights = {"a": 0.5, "b": 0.5}
        combined, record = combine(gradients, weights, "raw")
        self.assertEqual(combined[0].item(), 1.5)
        self.assertEqual(record["transformation"]["kind"], "identity")
        with self.assertRaises(ValueError):
            combine(gradients, weights, "unknown")


if __name__ == "__main__":
    unittest.main()
