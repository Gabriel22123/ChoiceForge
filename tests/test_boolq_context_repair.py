import unittest
from pathlib import Path
import sys

from scripts.boolq_context_repair_report import source_deltas
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_boolq_context_repair import (balanced_binary_weights, raw_distribution,
                                      token_admissible)


class BoolqContextRepairTests(unittest.TestCase):
    def test_binary_weights_equalize_class_mass_and_keep_mean_one(self):
        rows = [{"id": "n", "label": "no"},
                {"id": "y1", "label": "yes"}, {"id": "y2", "label": "yes"}]
        weights = balanced_binary_weights(rows)
        self.assertAlmostEqual(weights["n"], 1.5)
        self.assertAlmostEqual(weights["y1"] + weights["y2"], 1.5)
        self.assertAlmostEqual(sum(weights.values()) / len(rows), 1.0)

    def test_temperature_inversion_recovers_raw_distribution(self):
        raw = {"no": 0.25, "yes": 0.75}
        temperature = 2.0
        scaled = {key: value ** (1 / temperature) for key, value in raw.items()}
        total = sum(scaled.values())
        recovered = raw_distribution(
            {key: value / total for key, value in scaled.items()}, temperature)
        for key in raw:
            self.assertAlmostEqual(recovered[key], raw[key], places=12)

    def test_token_filter_uses_only_rendered_request(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                return text.split()
        request = {"task": "answer", "context": "short evidence", "choices": [
            {"id": "no", "description": "No"}, {"id": "yes", "description": "Yes"},
        ]}
        self.assertTrue(token_admissible(Tokenizer(), request, 20))
        self.assertFalse(token_admissible(Tokenizer(), request, 2))

    def test_source_delta_requires_the_same_sources(self):
        candidate = {"by_source": {"a": {"accuracy": .8}}}
        control = {"by_source": {"a": {"accuracy": .5}}}
        self.assertAlmostEqual(source_deltas(candidate, control)["a"], .3)
        with self.assertRaisesRegex(ValueError, "source sets"):
            source_deltas(candidate, {"by_source": {"b": {"accuracy": .5}}})


if __name__ == "__main__":
    unittest.main()
