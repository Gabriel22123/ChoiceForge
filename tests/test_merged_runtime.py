import unittest

import torch

from decision_model.merged_adapter import build_merged_residual_adapter
from decision_model.merged_runtime import MergedDecisionModel


class FakeParent:
    def __init__(self):
        self.torch = torch
        self.device = torch.device("cpu")
        self.head = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))

    def train(self, mode):
        return None

    def encode(self, request):
        return request

    def features(self, requests):
        rows = []
        for request in requests:
            context = 1.0 if request["context"] != "Context intentionally removed." else 0.0
            rows.append(torch.tensor([[2.0 + context, 0.0, 0.0],
                                      [1.0, context, 0.0]]))
        return torch.stack(rows)


class MergedRuntimeTest(unittest.TestCase):
    def test_prediction_is_strict_and_decision_stable(self):
        adapter = build_merged_residual_adapter(3, 2)
        for parameter in adapter.parameters():
            torch.nn.init.zeros_(parameter)
        metadata = {"format": "choiceforge-merged-residual-v1",
                    "architecture": {"hidden_size": 3, "total_width": 2},
                    "projection_alphas": [0.5, 1.0]}
        model = MergedDecisionModel(FakeParent(), adapter, metadata)
        request = {"task": "Choose.", "context": "Useful context.", "choices": [
            {"id": "a", "description": "A"}, {"id": "b", "description": "B"}]}
        result = model.predict(request)
        self.assertEqual(result["choice_id"], "a")
        self.assertTrue(result["requires_review"])
        self.assertEqual(set(result), {"choice_id", "probabilities", "requires_review",
                                       "score_kind", "calibration"})


if __name__ == "__main__":
    unittest.main()
