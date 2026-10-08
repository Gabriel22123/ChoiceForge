from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from semantic_collapse_diagnostic import align_predictions, parameter_changes


class ParameterDiagnosticTests(unittest.TestCase):
    def test_probability_columns_align_by_meaning_across_permuted_requests(self):
        original = [{"label": "b", "probabilities": {"a": .2, "b": .8}},
                    {"label": "b", "probabilities": {"b": .7, "a": .3}}]
        labels, aligned = align_predictions(original)
        self.assertEqual(labels, ["a", "b"])
        self.assertEqual(list(aligned[1]["probabilities"].values()), [.3, .7])
        self.assertEqual(list(original[1]["probabilities"]), ["b", "a"])
        with self.assertRaises(ValueError):
            align_predictions([original[0], {"probabilities": {"a": .2, "c": .8}}])

    def test_distinguishes_updated_layers_from_unchanged_layers(self):
        before = {"encoder.layers.0.lora_B": torch.ones(2),
                  "encoder.layers.1.lora_B": torch.zeros(2), "head.weight": torch.zeros(1)}
        after = {key: value.clone() for key, value in before.items()}
        after["encoder.layers.0.lora_B"] += 1
        after["head.weight"] += 2
        result = parameter_changes(before, after)
        self.assertEqual(result["updated_layers"], 1)
        self.assertEqual(result["total_changed_parameters"], 3)
        self.assertEqual(result["groups"]["layer-0"]["relative_l2_change"], 1)
        self.assertIsNone(result["groups"]["layer-1"]["relative_l2_change"])

    def test_rejects_nonfinite_weights_or_missing_parameters(self):
        with self.assertRaises(ValueError):
            parameter_changes({"head.weight": torch.ones(1)}, {})
        with self.assertRaises(ValueError):
            parameter_changes({"head.weight": torch.ones(1)}, {"head.weight": torch.tensor([float("nan")])})
