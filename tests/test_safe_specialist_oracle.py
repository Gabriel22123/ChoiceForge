import importlib.util
import json
import unittest
from pathlib import Path

from decision_model.core import digest, load_rows


SCRIPT = Path(__file__).parents[1] / "scripts/run_safe_specialist_oracle.py"
SPEC = importlib.util.spec_from_file_location("run_safe_specialist_oracle", SCRIPT)
SAFE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(SAFE)


class SafeSpecialistOracleTests(unittest.TestCase):

    def test_frozen_protocol_covers_exact_rows_and_canonical_weights(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "configs/safe-specialist-oracle-v1.json").read_text())
        self.assertEqual(config["status"], "frozen_before_evaluation")
        data = root / config["data"]["path"]
        if not data.is_file():
            self.skipTest("local experiment artifact is excluded from the public source release")
        self.assertEqual(digest(data.read_bytes()), config["data"]["sha256"])
        rows = [row for row in load_rows(data)
                if row["split"] in config["data"]["splits"] and
                row["source"] in config["data"]["sources"]]
        self.assertEqual(len(rows), config["data"]["rows"])
        for endpoint in config["experts"].values():
            self.assertEqual(digest((root / endpoint["weights"]).read_bytes()),
                             endpoint["weights_sha256"])
        for path, expected in config["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_safe_choice_rejects_correctness_or_metric_regression(self):
        parent = {"correct": True, "nll": .4, "brier": .2,
                  "context_advantage": .1, "proper_cost": .5}
        good = {"correct": True, "nll": .3, "brier": .1,
                "context_advantage": .2, "proper_cost": .35}
        wrong = {**good, "correct": False}
        worse_nll = {**good, "nll": .5}
        worse_brier = {**good, "brier": .3}
        worse_context = {**good, "context_advantage": .0}
        choice, improvement = SAFE.choose_safe(parent, {"good": good})
        self.assertEqual(choice, "good")
        self.assertAlmostEqual(improvement, .15)
        for value in (wrong, worse_nll, worse_brier, worse_context):
            self.assertEqual(SAFE.choose_safe(parent, {"bad": value}), ("parent", 0.0))

    def test_path_statistics_matches_binary_probability_order(self):
        strong = SAFE.path_statistics([2.0, -2.0], [0.0, 0.0], 0)
        weak = SAFE.path_statistics([0.0, 0.0], [0.0, 0.0], 0)
        self.assertTrue(strong["correct"])
        self.assertLess(strong["nll"], weak["nll"])
        self.assertLess(strong["brier"], weak["brier"])
        self.assertGreater(strong["context_advantage"], weak["context_advantage"])


if __name__ == "__main__":
    unittest.main()
