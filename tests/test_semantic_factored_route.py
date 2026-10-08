import importlib.util
import json
import unittest
from pathlib import Path

import torch

from decision_model.core import digest


SCRIPT = Path(__file__).parents[1] / "scripts/run_semantic_factored_route_lofo.py"
SPEC = importlib.util.spec_from_file_location("run_semantic_factored_route_lofo", SCRIPT)
ROUTE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ROUTE)


class SemanticFactoredRouteTests(unittest.TestCase):

    def test_protocol_is_frozen_without_relaxing_v1_gates(self):
        root = Path(__file__).parents[1]
        semantic = json.loads((root / "configs/semantic-factored-route-lofo-v1.json").read_text())
        behavior = json.loads((root / "configs/factored-safe-route-lofo-v1.json").read_text())
        self.assertEqual(semantic["status"], "frozen_before_training")
        self.assertEqual(semantic["screening_rule"]["violation_rate_max"],
                         behavior["screening_rule"]["violation_rate_max"])
        self.assertEqual(semantic["screening_rule"]["safe_selection_precision_min"],
                         behavior["screening_rule"]["safe_selection_precision_min"])
        self.assertEqual(semantic["screening_rule"]["expert_selection_rate_min"],
                         behavior["screening_rule"]["expert_selection_rate_min"])
        for path, expected in semantic["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_combined_features_add_four_projection_blocks(self):
        rows = [{"id": "one"}]
        records = {"one": {"features": torch.arange(5, dtype=torch.float32)}}
        generator = torch.Generator().manual_seed(9)
        frozen = {"cases:one": (torch.randn(3, 8, generator=generator),
                                torch.randn(3, 8, generator=generator))}
        first, projection = ROUTE.combined_route_features(
            rows, "cases", records, frozen, 8, 4, 1701)
        second, _ = ROUTE.combined_route_features(
            rows, "cases", records,
            {"cases:one": (frozen["cases:one"][0][[2, 0, 1]],
                           frozen["cases:one"][1][[2, 0, 1]])}, 8, 4, 1701)
        self.assertEqual(tuple(projection.shape), (8, 4))
        self.assertEqual(len(first["one"]), 21)
        self.assertTrue(torch.allclose(first["one"], second["one"], atol=1e-6, rtol=0))


if __name__ == "__main__": unittest.main()
