import unittest

from scripts.architecture_comparison_report import decoder_default_checks


def metrics(accuracy, nll=0.5):
    return {"accuracy": accuracy, "nll": nll}


def fixtures(trained_delta=0.02, arc_delta=0.02):
    arms, arc = {}, {}
    for seed in (42, 43, 44):
        encoder_sources = {name: metrics(0.70, 0.60) for name in ("paws-wiki", "snli")}
        decoder_sources = {name: metrics(0.70 + trained_delta, 0.59) for name in ("paws-wiki", "snli")}
        for name in ("bandit", "clinc", "goemotions", "json-schema"):
            encoder_sources[name] = metrics(0.60, 0.70)
            decoder_sources[name] = metrics(0.61, 0.69)
        arms[f"seed{seed}-encoder"] = {"raw": {"by_source": encoder_sources},
                                       "audit": {"stable_all_orders": 24}}
        arms[f"seed{seed}-decoder"] = {"raw": {"by_source": decoder_sources},
                                       "audit": {"stable_all_orders": 24}}
        arc_encoder = {name: metrics(0.50) for name in ("arc-challenge", "arc-easy")}
        arc_decoder = {name: metrics(0.50 + arc_delta) for name in ("arc-challenge", "arc-easy")}
        arc[f"seed{seed}-encoder"] = {"raw": {"by_source": arc_encoder}}
        arc[f"seed{seed}-decoder"] = {"raw": {"by_source": arc_decoder}}
    return arms, arc


class ArchitectureDefaultRuleTest(unittest.TestCase):
    def test_all_conditions_can_pass(self):
        arms, arc = fixtures()
        checks = decoder_default_checks(arms, arc)
        self.assertTrue(all(checks.values()))

    def test_sealed_arc_failure_blocks_default(self):
        arms, arc = fixtures(arc_delta=-0.01)
        checks = decoder_default_checks(arms, arc)
        self.assertFalse(checks["arc_combined_accuracy_wins_at_least_two_seeds"])
        self.assertFalse(checks["arc_mean_accuracy_not_worse"])
        self.assertFalse(checks["eligible_as_default"])

    def test_single_trained_task_regression_blocks_default(self):
        arms, arc = fixtures()
        arms["seed42-decoder"]["raw"]["by_source"]["paws-wiki"]["accuracy"] = 0.68
        checks = decoder_default_checks(arms, arc)
        self.assertFalse(checks["no_trained_task_accuracy_loss_below_minus_1pp"])
        self.assertFalse(checks["eligible_as_default"])

    def test_generalization_regression_blocks_default(self):
        arms, arc = fixtures()
        for seed in (42, 43, 44):
            for source in ("bandit", "clinc", "goemotions", "json-schema"):
                arms[f"seed{seed}-decoder"]["raw"]["by_source"][source] = metrics(0.59, 0.71)
        checks = decoder_default_checks(arms, arc)
        self.assertFalse(checks["generalization_mean_accuracy_not_worse"])
        self.assertFalse(checks["generalization_mean_nll_not_worse"])
        self.assertFalse(checks["eligible_as_default"])


if __name__ == "__main__":
    unittest.main()
