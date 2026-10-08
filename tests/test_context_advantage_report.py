import unittest

from scripts.context_advantage_report import advancement


def passing_effects():
    return {
        "primary_combined_accuracy_by_seed": [.01, .02, -.001],
        "primary_accuracy_all": [.01, .02, .01, .02, 0, -.005],
        "primary_nll_all": [-.01] * 6,
        "retained_accuracy_all": [0] * 12,
        "retained_nll_all": [-.001] * 12,
        "mechanism_gain_by_seed": [.1, .2, -.01],
        "mechanism_shortfall_by_seed": [-.1, -.1, 0],
        "blind_accuracy_by_seed": [.01, .02, -.001],
        "blind_nll_by_seed": [-.01, -.01, 0],
        "blind_ambiguous_recall_by_seed": [0, .01, 0],
        "blind_complete_group_accuracy_by_seed": [0, .01, 0],
    }


class ContextAdvantageReportTest(unittest.TestCase):
    def test_all_preregistered_checks_can_pass(self):
        checks = advancement(passing_effects(), [True, True, True])
        self.assertTrue(checks["eligible_as_default"])

    def test_one_failed_check_blocks_advancement(self):
        effects = passing_effects()
        effects["primary_accuracy_all"][0] = -.011
        checks = advancement(effects, [True, True, True])
        self.assertFalse(checks["primary_no_seed_task_accuracy_loss_below_minus_1pp"])
        self.assertFalse(checks["eligible_as_default"])

    def test_audit_failure_blocks_advancement(self):
        checks = advancement(passing_effects(), [True, False, True])
        self.assertFalse(checks["all_treatment_audits_stable"])
        self.assertFalse(checks["eligible_as_default"])


if __name__ == "__main__":
    unittest.main()
