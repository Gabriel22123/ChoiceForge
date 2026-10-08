import importlib.util
import json
import unittest
from pathlib import Path

from decision_model.core import digest, load_rows


SCRIPT = Path(__file__).parents[1] / "scripts/run_specialist_cross_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_specialist_cross_matrix", SCRIPT)
MATRIX = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MATRIX)


class SpecialistCrossMatrixTests(unittest.TestCase):

    def test_frozen_protocol_covers_exact_rows_sources_and_weights(self):
        root = Path(__file__).parents[1]
        config = json.loads((root / "configs/specialist-cross-matrix-v1.json").read_text())
        self.assertEqual(config["status"], "frozen_before_evaluation")
        data = root / config["data"]["path"]
        if not data.is_file():
            self.skipTest("local experiment artifact is excluded from the public source release")
        self.assertEqual(digest(data.read_bytes()), config["data"]["sha256"])
        rows = [row for row in load_rows(data)
                if row["split"] in config["data"]["splits"] and
                row["source"] in config["data"]["sources"]]
        self.assertEqual(len(rows), config["data"]["rows"])
        self.assertEqual(set(row["source"] for row in rows), set(config["data"]["sources"]))
        for record in config["experts"].values():
            self.assertIn(record["canonical_seed"],
                          [endpoint["seed"] for endpoint in record["endpoints"]])
            for endpoint in record["endpoints"]:
                self.assertEqual(digest((root / endpoint["weights"]).read_bytes()),
                                 endpoint["weights_sha256"])
        for path, expected in config["source_files"].items():
            self.assertEqual(digest((root / path).read_bytes()), expected)

    def test_oracle_keeps_parent_on_ties_and_uses_highest_positive_utility(self):
        choices = MATRIX.choose_oracle({"b": [0.2, -0.1, 0.0],
                                        "a": [0.2, 0.3, 0.0]})
        self.assertEqual(choices, [("a", 0.2), ("a", 0.3), ("parent", 0.0)])

    def test_oracle_rejects_misaligned_or_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            MATRIX.choose_oracle({"a": [0.1], "b": [0.1, 0.2]})
        with self.assertRaisesRegex(ValueError, "at least one"):
            MATRIX.choose_oracle({})


if __name__ == "__main__":
    unittest.main()
