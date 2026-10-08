import unittest

from decision_model.outcome_evaluation import outcome_metrics


def row(row_id, rewards):
    choices = [{"id": chr(97 + index), "description": chr(65 + index)}
               for index in range(len(rewards))]
    outcomes = {choice["id"]: {"passed": int(reward * 4), "total": 4,
                               "reward": reward,
                               "categories": ["passed"] * int(reward * 4) +
                                             ["assertion"] * (4 - int(reward * 4))}
                for choice, reward in zip(choices, rewards)}
    best = max(range(len(rewards)), key=rewards.__getitem__)
    return {"id": row_id, "label": choices[best]["id"],
            "request": {"task": "choose", "context": row_id, "choices": choices},
            "candidate_outcomes": outcomes}


class OutcomeEvaluationTests(unittest.TestCase):
    def test_ties_are_optimal_and_expected_reward_uses_full_policy(self):
        rows = [row("r1", [1., 1., 0.]), row("r2", [1., .5])]
        result, records = outcome_metrics(rows, [[0., 2., 1.], [0., 1.]])
        self.assertEqual(result["optimal_set_hit_rate"], .5)
        self.assertEqual(result["mean_decision_regret"], .25)
        self.assertEqual(records[0]["regret"], 0.)
        self.assertEqual(records[1]["selected_reward"], .5)
        self.assertGreater(result["mean_policy_expected_reward"], .5)

    def test_rejects_misaligned_logits(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            outcome_metrics([row("r", [1., 0.])], [[1.]])


if __name__ == "__main__":
    unittest.main()
