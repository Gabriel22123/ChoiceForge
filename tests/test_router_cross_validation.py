import unittest

from decision_model.router_cross_validation import binary_route_metrics, make_lofo_plan


class RouterCrossValidationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"id": f"{source}-{index}", "source": source}
            for source in ("a", "b", "c") for index in range(4)
        ]

    def test_balanced_plan_has_one_row_per_training_source_and_no_holdout(self):
        plan = make_lofo_plan(self.rows, "c", 17, 5, source_balanced=True)
        lookup = {row["id"]: row["source"] for row in self.rows}
        self.assertEqual(plan, make_lofo_plan(
            self.rows, "c", 17, 5, source_balanced=True))
        for update in plan:
            sources = [lookup[row_id] for row_id in update["row_ids"]]
            self.assertEqual(set(sources), {"a", "b"})
            self.assertEqual(len(sources), 2)

    def test_row_weighted_plan_has_equal_compute_and_no_holdout(self):
        plan = make_lofo_plan(self.rows, "b", 23, 7, source_balanced=False)
        self.assertEqual(len(plan), 7)
        self.assertTrue(all(len(update["row_ids"]) == 2 for update in plan))
        self.assertTrue(all(not row_id.startswith("b-")
                            for update in plan for row_id in update["row_ids"]))

    def test_binary_metrics_keep_probability_information(self):
        metrics = binary_route_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 0.085)
        self.assertAlmostEqual(metrics["target_open_rate"], 0.5)

    def test_invalid_plan_and_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            make_lofo_plan(self.rows, "missing", 1, 2, source_balanced=True)
        with self.assertRaises(ValueError):
            binary_route_metrics([0, 2], [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
