import importlib.util
import unittest
from pathlib import Path


def load_script():
    path = Path(__file__).parents[1] / "scripts/executable_outcome_screen_report.py"
    spec = importlib.util.spec_from_file_location("outcome_report", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class ExecutableOutcomeReportTests(unittest.TestCase):
    def test_advancement_needs_mean_two_seeds_stability_reward_and_integrity(self):
        module = load_script()
        rule = {"seed_wins_required": 2, "maximum_single_seed_regret_regression": .02,
                "integrity_violations_allowed": 0}
        good = [{"seed": seed, "regret_delta_D_minus_R": delta,
                 "expected_reward_delta_D_minus_R": .01, "integrity_violations": 0}
                for seed, delta in zip((42, 43, 44), (-.03, -.01, .01))]
        self.assertTrue(module.advancement(good, rule)["advances_to_lora_validation"])
        bad = [dict(row) for row in good]; bad[-1]["regret_delta_D_minus_R"] = .03
        self.assertFalse(module.advancement(bad, rule)["advances_to_lora_validation"])
        bad = [dict(row) for row in good]; bad[0]["integrity_violations"] = 1
        self.assertFalse(module.advancement(bad, rule)["advances_to_lora_validation"])


if __name__ == "__main__":
    unittest.main()
