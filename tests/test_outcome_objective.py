import itertools
import unittest

import torch

from decision_model.outcome_objective import (
    behavior_policy, doubly_robust_loss, full_information_loss, ips_loo_loss,
    optimal_set_loss,
)


class OutcomeObjectiveTests(unittest.TestCase):
    def test_behavior_policy_has_explicit_support_and_ignores_padding(self):
        logits = torch.tensor([[1., -2., 99.], [.2, .3, .4]])
        policy = behavior_policy(logits, [2, 3], exploration=.2)
        self.assertEqual(policy[0, 2].item(), 0.)
        self.assertGreaterEqual(policy[0, :2].min().item(), .1)
        self.assertTrue(torch.allclose(policy.sum(-1), torch.ones(2)))

    def test_expected_ips_gradient_equals_full_information_gradient(self):
        base = torch.tensor([[.3, -.2, .7]])
        rewards = torch.tensor([[1., .25, .0]])
        behavior = behavior_policy(base, [3], exploration=.3).detach()[0]
        full_logits = base.clone().requires_grad_()
        full_information_loss(full_logits, rewards, [3]).backward()
        expected = torch.zeros_like(full_logits)
        for action in range(3):
            logits = base.clone().requires_grad_()
            loss = ips_loo_loss(
                logits, torch.tensor([[action]]), torch.tensor([[rewards[0, action]]]),
                torch.tensor([[behavior[action]]]), [3])
            loss.backward()
            expected += behavior[action] * logits.grad
        self.assertTrue(torch.allclose(expected, full_logits.grad, atol=1e-6))

    def test_loo_baseline_remains_unbiased_for_two_logged_actions(self):
        base = torch.tensor([[.3, -.2, .7]])
        rewards = torch.tensor([[1., .25, .0]])
        behavior = behavior_policy(base, [3], exploration=.3).detach()[0]
        full_logits = base.clone().requires_grad_()
        full_information_loss(full_logits, rewards, [3]).backward()
        expected = torch.zeros_like(full_logits)
        for first, second in itertools.product(range(3), repeat=2):
            logits = base.clone().requires_grad_()
            actions = torch.tensor([[first, second]])
            observed = torch.tensor([[rewards[0, first], rewards[0, second]]])
            propensities = torch.tensor([[behavior[first], behavior[second]]])
            ips_loo_loss(logits, actions, observed, propensities, [3]).backward()
            expected += behavior[first] * behavior[second] * logits.grad
        self.assertTrue(torch.allclose(expected, full_logits.grad, atol=1e-6))

    def test_doubly_robust_policy_gradient_is_exact_for_arbitrary_estimate(self):
        base = torch.tensor([[.3, -.2, .7]])
        rewards = torch.tensor([[1., .25, .0]])
        estimate = torch.tensor([[.1, .8, .4]], requires_grad=True)
        behavior = behavior_policy(base, [3], exploration=.3).detach()[0]
        full_logits = base.clone().requires_grad_()
        full_information_loss(full_logits, rewards, [3]).backward()
        expected = torch.zeros_like(full_logits)
        for action in range(3):
            logits = base.clone().requires_grad_()
            loss, _ = doubly_robust_loss(
                logits, estimate, torch.tensor([[action]]),
                torch.tensor([[rewards[0, action]]]), torch.tensor([[behavior[action]]]),
                [3], reward_model_weight=0)
            loss.backward(retain_graph=True)
            expected += behavior[action] * logits.grad
        self.assertTrue(torch.allclose(expected, full_logits.grad, atol=1e-6))

    def test_chosen_only_contract_rejects_zero_propensity_and_bad_action(self):
        logits = torch.zeros(1, 2)
        with self.assertRaisesRegex(ValueError, "behavior probabilities"):
            ips_loo_loss(logits, torch.tensor([[0]]), torch.tensor([[1.]]),
                         torch.tensor([[0.]]), [2])
        with self.assertRaisesRegex(ValueError, "outside"):
            ips_loo_loss(logits, torch.tensor([[2]]), torch.tensor([[1.]]),
                         torch.tensor([[.5]]), [2])

    def test_optimal_set_loss_rewards_probability_on_any_tied_best_candidate(self):
        rewards = torch.tensor([[1., 1., 0.]])
        first = optimal_set_loss(torch.tensor([[5., -5., 0.]]), rewards, [3])
        second = optimal_set_loss(torch.tensor([[-5., 5., 0.]]), rewards, [3])
        wrong = optimal_set_loss(torch.tensor([[-5., -5., 5.]]), rewards, [3])
        self.assertAlmostEqual(first.item(), second.item(), places=5)
        self.assertLess(first, wrong)


if __name__ == "__main__":
    unittest.main()
