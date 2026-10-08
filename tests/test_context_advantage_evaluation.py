import math
import unittest

from decision_model.context_advantage_evaluation import summarize_context_advantage


def row(identity, source, label="a"):
    return {
        "id": identity, "source": source, "label": label,
        "request": {"task": "t", "context": "c", "choices": [
            {"id": "a", "description": "A"}, {"id": "b", "description": "B"}]},
    }


class ContextAdvantageEvaluationTest(unittest.TestCase):
    def test_summarizes_gold_gain_and_margin(self):
        rows = [row("1", "x"), row("2", "y", "b")]
        records = [
            {"id": "1", "evidence_logits": [2.0, 0.0], "prior_logits": [0.0, 0.0]},
            {"id": "2", "evidence_logits": [0.0, 0.0], "prior_logits": [2.0, 0.0]},
        ]
        result = summarize_context_advantage(rows, records, gap=.2)
        expected = (-math.log1p(math.exp(-2)) + math.log(2) +
                    -math.log(2) + math.log1p(math.exp(2))) / 2
        self.assertAlmostEqual(result["overall"]["mean_gold_log_probability_gain"], expected)
        self.assertEqual(result["overall"]["positive_gain_rate"], 1.0)
        self.assertEqual(set(result["by_source"]), {"x", "y"})

    def test_rejects_mismatched_records_and_gap(self):
        with self.assertRaises(ValueError):
            summarize_context_advantage([row("1", "x")], [], gap=.2)
        with self.assertRaises(ValueError):
            summarize_context_advantage([row("1", "x")], [
                {"id": "1", "evidence_logits": [0, 0], "prior_logits": [0, 0]}], gap=-1)


if __name__ == "__main__":
    unittest.main()
