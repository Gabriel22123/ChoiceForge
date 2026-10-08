import importlib.util
from pathlib import Path
import unittest

import torch


PATH = Path(__file__).resolve().parents[1] / "scripts/run_scifact_final_evaluation.py"
SPEC = importlib.util.spec_from_file_location("scifact_final_evaluation_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


def row():
    return {"request": {"task": "Choose.", "context": "Evidence.", "choices": [
        {"id": "a", "description": "A"}, {"id": "b", "description": "B"}]},
        "label": "a"}


class SciFactFinalEvaluationTest(unittest.TestCase):
    def test_prediction_uses_strict_review_contract(self):
        prediction = MODULE.build_prediction(row(), torch.tensor([2.0, 1.0]))
        self.assertEqual(prediction["choice_id"], "a")
        self.assertTrue(prediction["requires_review"])
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0)

    def test_metrics_include_proper_cost_and_context(self):
        value = MODULE.score_paths([row()], [[2.0, 1.0]], [[0.0, 0.0]], 0.5)
        self.assertEqual(value["accuracy"], 1.0)
        self.assertGreater(value["context_advantage"], 0.0)
        self.assertAlmostEqual(value["proper_cost"],
                               value["cross_entropy"] + 0.5 * value["brier"])

    def test_permutation_audit_accepts_candidate_wise_projection(self):
        parent = torch.tensor([2.0, 1.0])
        portfolio = torch.tensor([1.8, 1.2])
        self.assertTrue(MODULE.permutation_audit(
            parent, torch.tensor([0.2, 0.1]), portfolio,
            torch.tensor([0.1, 0.2]), [0.125, 0.5, 1.0]))


if __name__ == "__main__":
    unittest.main()
