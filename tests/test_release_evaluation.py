import importlib.util
import math
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "scripts/release_evaluation.py"
spec = importlib.util.spec_from_file_location("release_evaluation", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{
            "id": "a", "source": "fresh", "label": "yes",
            "request": {"task": "decide", "context": "evidence",
                        "choices": [{"id": "yes", "description": "yes"},
                                    {"id": "no", "description": "no"}]},
        }]

    def test_context_advantage_uses_gold_probability(self):
        summary, points = module.context_advantage(self.rows, [{
            "id": "a", "evidence_logits": [2.0, 0.0], "prior_logits": [0.0, 0.0],
        }])
        self.assertGreater(summary["mean"], 0)
        self.assertEqual(summary["by_source"]["fresh"]["positive"], 1)
        expected = math.log((math.exp(2) / (math.exp(2) + 1)) / 0.5)
        self.assertAlmostEqual(points[0]["gold_log_probability_advantage"], expected)

    def test_cross_seed_js_zero_for_equal_distributions(self):
        common = [{"id": "a", "prior_logits": [1.0, -1.0]}]
        summary, _ = module.cross_seed_null_js({1701: common, 1702: common, 1703: common})
        self.assertAlmostEqual(summary["mean_js"], 0)

    def test_cross_seed_js_detects_prior_disagreement(self):
        summary, _ = module.cross_seed_null_js({
            "left": [{"id": "a", "prior_logits": [8.0, -8.0]}],
            "right": [{"id": "a", "prior_logits": [-8.0, 8.0]}],
        })
        self.assertGreater(summary["mean_js"], 0.69)

    def test_cross_seed_rejects_mismatched_rows(self):
        with self.assertRaises(ValueError):
            module.cross_seed_null_js({
                1: [{"id": "a", "prior_logits": [0.0, 0.0]}],
                2: [{"id": "b", "prior_logits": [0.0, 0.0]}],
            })


if __name__ == "__main__":
    unittest.main()
