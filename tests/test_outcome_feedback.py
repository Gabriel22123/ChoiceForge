import unittest

from decision_model.outcome_feedback import consolidate_feedback


class OutcomeFeedbackTests(unittest.TestCase):
    def rows(self):
        return [{"id": "r", "request": {"choices": [{"id": "a"}, {"id": "b"}, {"id": "c"}]}}]

    def test_consolidates_only_observed_actions_and_detects_completion(self):
        partial = [{"row": "r", "actions": [0, 2, 0], "rewards": [1., 0., 1.]}]
        complete, missing, report = consolidate_feedback(self.rows(), partial)
        self.assertEqual(complete, {})
        self.assertEqual(missing, {"r": [1]})
        self.assertEqual(report["missing_candidate_pairs"], 1)
        logs = partial + [{"row": "r", "actions": [1], "rewards": [.5]}]
        complete, missing, report = consolidate_feedback(self.rows(), logs)
        self.assertEqual(complete, {"r": [1., .5, 0.]})
        self.assertEqual(missing, {})
        self.assertEqual(report["complete_rows"], 1)

    def test_rejects_inconsistent_repeated_outcomes(self):
        logs = [{"row": "r", "actions": [0, 0], "rewards": [1., 0.]}]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            consolidate_feedback(self.rows(), logs)

    def test_interface_does_not_need_hidden_outcomes(self):
        rows = self.rows()
        rows[0]["candidate_outcomes"] = {"secret": "not inspected"}
        complete, missing, _ = consolidate_feedback(
            rows, [{"row": "r", "actions": [0], "rewards": [1.]}])
        self.assertEqual(complete, {})
        self.assertEqual(missing["r"], [1, 2])


if __name__ == "__main__":
    unittest.main()
