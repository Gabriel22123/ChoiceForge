import unittest

import torch

from decision_model.objective import proper_cost, proper_loss
from decision_model.training_objective import (TrainingObjective, context_advantage_loss,
                                               distillation_kl_loss,
                                               ranked_probability_cost)


class TrainingObjectiveTests(unittest.TestCase):
    def test_distillation_kl_is_zero_at_teacher_and_respects_weights(self):
        teacher = torch.tensor([[0.8, 0.2, 0.0], [0.1, 0.3, 0.6]])
        logits = torch.log(torch.tensor([[0.8, 0.2, 1.0], [0.1, 0.3, 0.6]]))
        unweighted = distillation_kl_loss(logits, teacher, [2, 3])
        self.assertAlmostEqual(unweighted.item(), 0.0, places=6)
        changed = logits.clone()
        changed[0, :2] = torch.log(torch.tensor([0.2, 0.8]))
        weighted = distillation_kl_loss(
            changed, teacher, [2, 3], sample_weights=torch.tensor([2.0, 0.5]))
        self.assertGreater(weighted.item(), 0.0)

    def test_distillation_kl_rejects_bad_padding_and_shape(self):
        logits = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "match padded logits"):
            distillation_kl_loss(logits, torch.ones(1, 2), [2])
        with self.assertRaisesRegex(ValueError, "padded mass"):
            distillation_kl_loss(logits, torch.tensor([[0.5, 0.4, 0.1]]), [2])

    def test_distillation_kl_can_mask_rows_without_teacher_targets(self):
        logits = torch.tensor([[0.0, 0.0], [2.0, -2.0]], requires_grad=True)
        targets = torch.tensor([[0.5, 0.5], [0.0, 0.0]])
        loss = distillation_kl_loss(
            logits, targets, [2, 2], active_mask=torch.tensor([1.0, 0.0]))
        self.assertAlmostEqual(loss.item(), 0.0, places=6)
        loss.backward()
        self.assertEqual(logits.grad[1].abs().sum().item(), 0.0)
        with self.assertRaisesRegex(ValueError, "binary batch vector"):
            distillation_kl_loss(
                logits.detach(), targets, [2, 2], active_mask=torch.tensor([1.0, 0.5]))

    def test_legacy_clean_loss_is_unchanged(self):
        logits = torch.tensor([[.2, -.3, -1e4], [.1, .2, -.1]], requires_grad=True)
        targets = torch.tensor([0, 2])
        loss, risk = TrainingObjective({"brier_weight": .5})(logits, targets, [2, 3])
        expected = proper_loss(logits, targets, .5)
        self.assertTrue(torch.equal(loss, expected))
        self.assertTrue(torch.equal(risk, expected.detach()))
        self.assertFalse(risk.requires_grad)

    def test_clean_soft_target_uses_proper_distribution_loss(self):
        logits = torch.tensor([[.2, -.3, -1e4], [.1, .2, -.1]], requires_grad=True)
        targets = torch.tensor([[.7, .3, 0.], [.2, .3, .5]])
        loss, risk = TrainingObjective({"brier_weight": .5})(logits, targets, [2, 3])
        expected = proper_loss(logits, targets, .5)
        self.assertTrue(torch.equal(loss, expected))
        self.assertTrue(torch.equal(risk, expected.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_clean_sample_weights_apply_per_example_without_batch_renormalization(self):
        logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
        targets = torch.tensor([0, 0])
        weights = torch.tensor([1.5, 0.5])
        loss, risk = TrainingObjective({"brier_weight": .5})(
            logits, targets, [2, 2], sample_weights=weights)
        expected = (proper_cost(logits, targets, .5) * weights).mean()
        self.assertTrue(torch.equal(loss, expected))
        self.assertTrue(torch.equal(risk, expected.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_sample_weights_reject_unsupported_or_invalid_combinations(self):
        logits = torch.zeros(2, 2)
        targets = torch.tensor([0, 1])
        invalid = (torch.tensor([1.0]), torch.tensor([1.0, 0.0]),
                   torch.tensor([1.0, float("nan")]))
        for weights in invalid:
            with self.assertRaisesRegex(ValueError, "Sample weights"):
                TrainingObjective({})(logits, targets, [2, 2], sample_weights=weights)
        with self.assertRaisesRegex(ValueError, "ranked-probability"):
            TrainingObjective({"ranked_probability_weight": 1.0})(
                logits, targets, [2, 2], ["score", "score"],
                sample_weights=torch.ones(2))
        with self.assertRaisesRegex(ValueError, "clean objective"):
            TrainingObjective({"gradient_estimator": "pathwise"})(
                logits, targets, [2, 2], sample_weights=torch.ones(2))

    def test_soft_targets_support_loo_score_estimator_and_reject_padded_mass(self):
        logits=torch.zeros(1,3)
        objective=TrainingObjective({"gradient_estimator":"score_function","noise_samples":8})
        train_logits=logits.clone().requires_grad_()
        loss,risk=objective(train_logits,torch.tensor([[.6,.4,0.]]),[2],["boolean"])
        loss.backward()
        self.assertTrue(torch.isfinite(train_logits.grad).all())
        self.assertFalse(risk.requires_grad)
        with self.assertRaisesRegex(ValueError,"padded mass"):
            TrainingObjective({})(logits,torch.tensor([[.5,.4,.1]]),[2])

    def test_ordered_rps_and_clean_anchor(self):
        target=torch.tensor([[.1,.2,.7]])
        aligned=ranked_probability_cost(target.log(),target)
        reversed_cost=ranked_probability_cost(target.flip(-1).log(),target)
        self.assertLess(aligned.item(),reversed_cost.item())
        logits=torch.tensor([[.2,.1,-.3]],requires_grad=True)
        plain=TrainingObjective({"ranked_probability_weight":0})(logits,target,[3],["score"])[0]
        typed=TrainingObjective({"ranked_probability_weight":1})(logits,target,[3],["score"])[0]
        self.assertGreater(typed.item(),plain.item())
        with self.assertRaisesRegex(ValueError,"decision_type"):
            TrainingObjective({"ranked_probability_weight":1})(logits,target,[3])

    def test_ordered_rps_undoes_candidate_presentation_permutation(self):
        config = {"ranked_probability_weight": 1.0, "brier_weight": 0.5}
        semantic_logits = torch.tensor([[.7, -.2, 1.1]], requires_grad=True)
        semantic_target = torch.tensor([[.1, .2, .7]])
        reference, _ = TrainingObjective(config)(
            semantic_logits, semantic_target, [3], ["score"])
        order = [2, 0, 1]
        permuted_logits = semantic_logits[:, order].detach().requires_grad_()
        permuted_target = semantic_target[:, order]
        actual, _ = TrainingObjective(config)(
            permuted_logits, permuted_target, [3], ["score"], candidate_orders=[order])
        self.assertAlmostEqual(reference.item(), actual.item(), places=6)
        with self.assertRaisesRegex(ValueError, "semantic candidate order"):
            TrainingObjective(config)(permuted_logits, permuted_target, [3], ["score"],
                                      candidate_orders=[[0, 0, 2]])

    def test_noisy_rps_undoes_candidate_presentation_permutation(self):
        for estimator in ("pathwise", "score_function"):
            config = {"gradient_estimator": estimator, "ranked_probability_weight": 1.0,
                      "noise_seed": 29, "noise_samples": 8}
            logits = torch.tensor([[.7, -.2, 1.1]], requires_grad=True)
            target = torch.tensor([[.1, .2, .7]])
            reference, reference_risk = TrainingObjective(config)(
                logits, target, [3], ["score"])
            order = [2, 0, 1]
            actual, actual_risk = TrainingObjective(config)(
                logits[:, order], target[:, order], [3], ["score"], candidate_orders=[order])
            self.assertAlmostEqual(reference.item(), actual.item(), places=5)
            self.assertAlmostEqual(reference_risk.item(), actual_risk.item(), places=5)

    def test_noise_does_not_consume_dropout_rng(self):
        objective = TrainingObjective({"gradient_estimator": "pathwise", "noise_seed": 77})
        logits = torch.tensor([[.1, .2]], requires_grad=True)
        state = torch.get_rng_state().clone()
        objective(logits, torch.tensor([0]), [2])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        other = TrainingObjective({"gradient_estimator": "score_function", "noise_seed": 77})
        torch.randn(13)  # Unrelated global draws must not change the noise plan.
        other(logits, torch.tensor([0]), [2])
        self.assertEqual(objective.record()["noise_plan_sha256"], other.record()["noise_plan_sha256"])

    def test_projected_estimators_agree_and_exclude_padding(self):
        config = {"noise_samples": 32768, "noise_seed": 41, "center_score_logits": True}
        logits = torch.tensor([[.2, -.1, 20.], [.1, .2, -.3]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([1, 2])
        path = TrainingObjective(dict(config, gradient_estimator="pathwise"))
        score = TrainingObjective(dict(config, gradient_estimator="score_function"))
        path_loss, path_risk = path(logits, targets, [2, 3])
        score_loss, score_risk = score(logits, targets, [2, 3])
        path_grad = torch.autograd.grad(path_loss, logits)[0]
        score_grad = torch.autograd.grad(score_loss, logits)[0]
        self.assertTrue(torch.equal(path_risk, score_risk))
        self.assertLess((path_grad - score_grad).abs().max().item(), .02)
        self.assertEqual(score_grad[0, 2].item(), 0)
        self.assertEqual(path_grad[0, 2].item(), 0)
        self.assertLess(score_grad.sum(-1).abs().max().item(), 1e-12)
        self.assertNotAlmostEqual(score_loss.item(), score_risk.item(), places=3)

    def test_rejects_invalid_noise_and_padding_target(self):
        for config in ({"noise_sigma": 0}, {"noise_samples": 1}, {"gradient_estimator": "unknown"},
                       {"clean_anchor_weight":1}):
            with self.assertRaises(ValueError):
                TrainingObjective(config)
        with self.assertRaisesRegex(ValueError, "Target"):
            TrainingObjective({})(torch.zeros(1, 3), torch.tensor([2]), [2])

    def test_context_advantage_penalizes_missing_evidence_gain(self):
        evidence = torch.tensor([[0.0, 0.0, 20.0], [1.5, 0.0, -1.0]], requires_grad=True)
        null = torch.tensor([[0.0, 0.0, -20.0], [1.5, 0.0, -1.0]], requires_grad=True)
        targets = torch.tensor([0, 0])
        loss = context_advantage_loss(evidence, null, targets, [2, 3], gap=.2)
        self.assertGreater(loss.item(), 0)
        evidence_grad, null_grad = torch.autograd.grad(loss, (evidence, null), allow_unused=True)
        self.assertIsNotNone(evidence_grad)
        self.assertIsNone(null_grad)
        self.assertEqual(evidence_grad[0, 2].item(), 0)

    def test_null_teacher_preserves_null_branch_without_changing_margin_gradient(self):
        evidence = torch.tensor([[0.0, 0.0]], requires_grad=True)
        null = torch.tensor([[1.0, -1.0]], requires_grad=True)
        target = torch.tensor([0])
        margin = context_advantage_loss(evidence, null, target, [2], gap=.2)
        null_teacher = torch.tensor([[0.5, 0.5]])
        preserve = distillation_kl_loss(null, null_teacher, [2])
        evidence_grad, null_grad = torch.autograd.grad(
            margin + preserve, (evidence, null))
        margin_evidence_grad = torch.autograd.grad(
            context_advantage_loss(evidence, null.detach(), target, [2], gap=.2),
            evidence)[0]
        self.assertTrue(torch.allclose(evidence_grad, margin_evidence_grad))
        self.assertGreater(null_grad.abs().sum().item(), 0)

    def test_context_advantage_is_zero_after_sufficient_gain(self):
        evidence = torch.tensor([[3.0, 0.0]], requires_grad=True)
        null = torch.tensor([[0.0, 0.0]])
        value = context_advantage_loss(evidence, null, torch.tensor([0]), [2], gap=.2)
        self.assertEqual(value.item(), 0)

    def test_context_advantage_accepts_probability_target(self):
        evidence=torch.tensor([[2.,0.]],requires_grad=True)
        null=torch.tensor([[0.,0.]])
        value=context_advantage_loss(evidence,null,torch.tensor([[.75,.25]]),[2],gap=.1)
        self.assertTrue(torch.isfinite(value))

    def test_context_advantage_applies_the_same_sample_weights(self):
        evidence = torch.tensor([[0., 0.], [0., 0.]], requires_grad=True)
        null = torch.tensor([[0., 0.], [2., 0.]])
        targets = torch.tensor([0, 0])
        weights = torch.tensor([1.5, .5])
        unweighted = context_advantage_loss(evidence, null, targets, [2, 2], gap=.2)
        weighted = context_advantage_loss(
            evidence, null, targets, [2, 2], gap=.2, sample_weights=weights)
        self.assertNotEqual(unweighted.item(), weighted.item())

    def test_context_advantage_rejects_invalid_inputs(self):
        with self.assertRaisesRegex(ValueError, "gap"):
            context_advantage_loss(torch.zeros(1, 2), torch.zeros(1, 2),
                                   torch.tensor([0]), [2], gap=-1)
        with self.assertRaisesRegex(ValueError, "differ"):
            context_advantage_loss(torch.zeros(1, 2), torch.zeros(1, 3),
                                   torch.tensor([0]), [2])


if __name__ == "__main__":
    unittest.main()
