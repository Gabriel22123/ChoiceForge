import unittest

import torch

from decision_model.prior_evaluation import (
    adjusted_logits, candidate_marginal_gap, collect_evidence_and_prior_logits,
    evaluate_with_temperature, fit_and_evaluate,
)


def row(identity, context, label="b"):
    return {"id": identity, "label": label, "source": "synthetic", "request": {
        "task": "Choose evidence-supported candidate", "context": context,
        "choices": [{"id": "a", "description": "popular"},
                    {"id": "b", "description": "supported"}],
    }}


class FakeJudge:
    def __init__(self):
        self.torch = torch
        self.calls = 0

    def train(self, mode=True):
        return None

    def encode(self, request):
        return request

    def logits(self, requests):
        self.calls += 1
        values = []
        for request in requests:
            evidence = 2.0 if request["context"].startswith("supports") else 0.0
            values.append([1.0, evidence])
        return torch.tensor(values)


class PriorEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.rows = [row("one", "supports b"), row("two", "supports b again")]

    def test_collection_caches_identical_null_request(self):
        judge = FakeJudge()
        records, summary = collect_evidence_and_prior_logits(judge, self.rows)
        self.assertEqual(judge.calls, 3)
        self.assertEqual(summary, {"rows": 2, "unique_prior_requests": 1})
        self.assertEqual([record["id"] for record in records], ["one", "two"])

    def test_adjustment_removes_candidate_prior(self):
        records = [
            {"id": "one", "evidence_logits": [1.0, 2.0], "prior_logits": [1.0, 0.0]},
            {"id": "two", "evidence_logits": [1.0, 2.0], "prior_logits": [1.0, 0.0]},
        ]
        self.assertEqual(adjusted_logits(self.rows, records, 1.0), [[0.5, 2.5], [0.5, 2.5]])

    def test_fit_and_frozen_temperature_evaluation(self):
        judge = FakeJudge()
        records, _ = collect_evidence_and_prior_logits(judge, self.rows)
        fitted = fit_and_evaluate(self.rows, records, 0.5)
        frozen = evaluate_with_temperature(
            self.rows, records, 0.5, fitted["calibration"]["temperature"])
        self.assertEqual(fitted["raw"], frozen["raw"])
        self.assertEqual(fitted["temperature_scaled"], frozen["temperature_scaled"])
        self.assertEqual(fitted["predictions"], frozen["predictions"])

    def test_rejects_wrong_record_identity(self):
        records = [{"id": "other", "evidence_logits": [0, 1], "prior_logits": [0, 1]}]
        with self.assertRaisesRegex(ValueError, "IDs"):
            adjusted_logits(self.rows, records, 1.0)

    def test_candidate_marginal_gap(self):
        predictions = [
            {"id": "one", "label": "b", "probabilities": {"a": 0.9, "b": 0.1}},
            {"id": "two", "label": "b", "probabilities": {"a": 0.8, "b": 0.2}},
        ]
        value = candidate_marginal_gap(self.rows, predictions)["synthetic"]
        self.assertEqual(value["truth"], {"b": 2})
        self.assertEqual(value["predicted"], {"a": 2})
        self.assertEqual(value["total_variation_gap"], 1.0)


if __name__ == "__main__":
    unittest.main()
