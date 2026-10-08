import unittest

from scripts.functional_retention_screen_report import compare
from scripts.run_functional_retention_screen import raw_distribution, same_unique_ids


class FunctionalRetentionScreenTests(unittest.TestCase):
    def test_selected_subset_identity_ignores_order_but_rejects_duplicates(self):
        self.assertTrue(same_unique_ids(["a", "b"], ["b", "a"]))
        self.assertFalse(same_unique_ids(["a", "a"], ["a", "b"]))

    def test_temperature_inversion_recovers_raw_distribution(self):
        raw = {"a": 0.8, "b": 0.2}
        temperature = 2.0
        scaled = {key: value ** (1 / temperature) for key, value in raw.items()}
        total = sum(scaled.values())
        recovered = raw_distribution(
            {key: value / total for key, value in scaled.items()}, temperature)
        for key in raw:
            self.assertAlmostEqual(recovered[key], raw[key], places=12)

    def test_compare_enforces_combined_source_and_nll_rules(self):
        control = {"accuracy": .7, "nll": .8,
                   "by_source": {"a": {"accuracy": .7}, "b": {"accuracy": .7}}}
        expanded = {"accuracy": .75, "nll": .7,
                    "by_source": {"a": {"accuracy": .75}, "b": {"accuracy": .75}}}
        candidate = {"accuracy": .746, "nll": .705,
                     "by_source": {"a": {"accuracy": .69}, "b": {"accuracy": .8}}}
        rule = {
            "seen_combined_accuracy_vs_parent_expanded_min_delta": -.005,
            "seen_each_source_accuracy_vs_parent_control_min_delta": -.02,
            "seen_nll_vs_parent_expanded_max_delta": .01,
        }
        _, checks = compare(candidate, control, expanded, rule, "seen")
        self.assertTrue(all(checks.values()))


if __name__ == "__main__":
    unittest.main()
