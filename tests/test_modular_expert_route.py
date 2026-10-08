import importlib.util
import json
import unittest
from pathlib import Path

import torch

from decision_model.core import digest


SCRIPT = Path(__file__).parents[1] / "scripts/run_modular_expert_route.py"
SPEC = importlib.util.spec_from_file_location("run_modular_expert_route", SCRIPT)
ROUTE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ROUTE)


class ModularExpertRouteTests(unittest.TestCase):
    def test_protocol_freezes_independent_width_and_original_gates(self):
        root = Path(__file__).parents[1]
        modular = json.loads((root / "configs/modular-expert-route-v1.json").read_text())
        shared = json.loads((root / "configs/extended-semantic-consensus-route-v1.json").read_text())
        self.assertEqual(modular["status"], "frozen_before_training")
        self.assertEqual(modular["module_input_width"], 280)
        for key in ("violation_rate_max", "safe_selection_precision_min",
                    "expert_selection_rate_min", "proper_gain_min"):
            self.assertEqual(modular["screening_rule"][key], shared["screening_rule"][key])
        for path, expected in modular["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_feature_maps_are_fixed_width_and_isolated(self):
        names = ("a", "b")
        raw = {"row": torch.arange(7 + 2 * 17 + 8, dtype=torch.float32)}
        maps = ROUTE.expert_feature_maps(raw, names, 8)
        self.assertEqual(len(maps["a"]["row"]), 32)
        changed = {"row": raw["row"].clone()}; changed["row"][24:41] += 100
        changed_maps = ROUTE.expert_feature_maps(changed, names, 8)
        self.assertTrue(torch.equal(maps["a"]["row"], changed_maps["a"]["row"]))
        self.assertFalse(torch.equal(maps["b"]["row"], changed_maps["b"]["row"]))


if __name__ == "__main__": unittest.main()
