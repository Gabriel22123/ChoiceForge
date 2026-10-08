import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path

from decision_model.core import digest, load_rows


SCRIPT = Path(__file__).parents[1] / "scripts/run_safe_multi_expert_route_lofo.py"
SPEC = importlib.util.spec_from_file_location("run_safe_multi_expert_route_lofo", SCRIPT)
ROUTE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ROUTE)


class SafeMultiExpertRouteTests(unittest.TestCase):

    def test_frozen_protocol_has_exact_fold_counts_and_weights(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "configs/safe-multi-expert-route-lofo-v1.json").read_text())
        self.assertEqual(config["status"], "frozen_before_training")
        data = root / config["data"]["path"]
        if not data.is_file():
            self.skipTest("local experiment artifact is excluded from the public source release")
        self.assertEqual(digest(data.read_bytes()), config["data"]["sha256"])
        rows = [row for row in load_rows(data) if row["split"] in ("train", "validation") and
                row["source"] in config["data"]["sources"]]
        counts = Counter(f"{row['source']}|{row['split']}" for row in rows)
        self.assertEqual(dict(counts), config["data"]["counts"])
        for endpoint in config["experts"].values():
            self.assertEqual(digest((root / endpoint["weights"]).read_bytes()),
                             endpoint["weights_sha256"])
        for path, expected in config["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_source_balanced_plan_excludes_heldout_and_is_deterministic(self):
        rows = [{"id": f"{source}-{index}", "source": source}
                for source in ("a", "b", "held") for index in range(3)]
        first = ROUTE.source_balanced_plan(rows, "held", 42, 3, 4)
        second = ROUTE.source_balanced_plan(list(reversed(rows)), "held", 42, 3, 4)
        self.assertEqual(first, second)
        self.assertTrue(all(not row_id.startswith("held-")
                            for batch in first for row_id in batch["row_ids"]))

    def test_class_weights_are_finite_capped_and_zero_for_missing_path(self):
        weights = ROUTE.class_weights([0, 0, 0, 1], 3, 3.0)
        self.assertEqual(float(weights[2]), 0.0)
        self.assertLessEqual(float(weights.max()), 3.0)
        self.assertAlmostEqual(float(weights[:2].mean()), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
