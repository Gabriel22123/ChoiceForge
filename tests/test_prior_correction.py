import unittest

import torch

from decision_model.prior_correction import (NULL_CONTEXT, STRENGTH_GRID, corrected_logits,
                                             predict_with_prior_correction,
                                             prior_only_request, select_shared_strength)


class FakeJudge:
    def __init__(self):
        self.torch = torch
        self.mode = None

    def train(self, mode=True):
        self.mode = mode

    def encode(self, request):
        return request

    def logits(self, encoded):
        rows = []
        for request in encoded:
            values = []
            for choice in request["choices"]:
                prior = {"popular": 2.0, "supported": 0.0}[choice["id"]]
                evidence = 1.5 if request["context"] != NULL_CONTEXT and choice["id"] == "supported" else 0.0
                values.append(prior + evidence)
            rows.append(values)
        return torch.tensor(rows)


class CandidatePriorCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.request = {"task": "Choose the supported option.", "context": "Evidence supports B.", "choices": [
            {"id": "popular", "description": "A commonly selected answer."},
            {"id": "supported", "description": "The answer supported by B."},
        ]}

    def test_prior_request_keeps_candidates_and_removes_evidence(self):
        value = prior_only_request(self.request)
        self.assertEqual(value["task"], self.request["task"])
        self.assertEqual(value["choices"], self.request["choices"])
        self.assertEqual(value["context"], NULL_CONTEXT)

    def test_subtraction_removes_candidate_only_preference(self):
        judge = FakeJudge()
        raw = judge.logits([judge.encode(self.request)])[0]
        corrected = corrected_logits(judge, self.request)
        self.assertEqual(raw.argmax().item(), 0)
        self.assertGreater(corrected[1].item() - corrected[0].item(), raw[1].item() - raw[0].item())
        result = predict_with_prior_correction(judge, self.request)
        self.assertEqual(result["choice_id"], "supported")
        self.assertTrue(result["requires_review"])
        self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=6)

    def test_candidate_permutation_is_equivariant(self):
        judge = FakeJudge()
        left = predict_with_prior_correction(judge, self.request)["probabilities"]
        permuted = dict(self.request, choices=list(reversed(self.request["choices"])))
        right = predict_with_prior_correction(judge, permuted)["probabilities"]
        self.assertEqual(set(left), set(right))
        for key in left:
            self.assertAlmostEqual(left[key], right[key], places=7)

    def test_invalid_controls_are_rejected(self):
        judge = FakeJudge()
        for strength in (-1, float("nan"), True):
            with self.assertRaisesRegex(ValueError, "strength"):
                corrected_logits(judge, self.request, strength)
        with self.assertRaisesRegex(ValueError, "Temperature"):
            predict_with_prior_correction(judge, self.request, temperature=0)

    def test_shared_strength_uses_mean_validation_nll(self):
        values = {
            42: {strength: 0.8 for strength in STRENGTH_GRID},
            43: {strength: 0.8 for strength in STRENGTH_GRID},
        }
        values[42][0.5] = 0.5
        values[43][0.5] = 0.7
        result = select_shared_strength(values)
        self.assertEqual(result["selected_strength"], 0.5)
        self.assertAlmostEqual(result["mean_validation_nll"][0.5], 0.6)

    def test_shared_strength_tie_prefers_smaller_value(self):
        values = {seed: {strength: 1.0 for strength in STRENGTH_GRID} for seed in (42, 43)}
        self.assertEqual(select_shared_strength(values)["selected_strength"], 0.0)

    def test_shared_strength_rejects_missing_or_invalid_grid(self):
        with self.assertRaises(ValueError):
            select_shared_strength({42: {0.0: 1.0}, 43: {0.0: 1.0}})
        with self.assertRaises(ValueError):
            select_shared_strength({42: {strength: 1.0 for strength in STRENGTH_GRID}})


if __name__ == "__main__":
    unittest.main()
