import json
import math
import unittest
from pathlib import Path

from decision_model.core import digest, load_rows


ROOT = Path(__file__).parents[1]
CONFIGS = (
    "configs/arc-specialist-v1.json",
    "configs/snli-specialist-v1.json",
    "configs/paws-wiki-specialist-v1.json",
    "configs/typed-decisions-specialist-v1.json",
    "configs/clinc-specialist-v1.json",
    "configs/json-schema-specialist-v1.json",
    "configs/truthfulqa-specialist-v1.json",
)


class SpecialistProtocolTests(unittest.TestCase):

    def test_consumed_source_protocols_are_frozen_and_self_consistent(self):
        evidence_paths, report_paths = set(), set()
        for relative in CONFIGS:
            config = json.loads((ROOT / relative).read_text())
            self.assertEqual(config["status"], "frozen_before_training")
            self.assertEqual(config["training"]["seeds"], [42, 43, 44])
            train_path = ROOT / config["data"]["train"]
            validation_path = ROOT / config["data"]["validation"]
            if not train_path.is_file() or not validation_path.is_file():
                self.skipTest("local experiment artifacts are excluded from the public source release")
            self.assertEqual(digest(train_path.read_bytes()), config["data"]["train_sha256"])
            self.assertEqual(digest(validation_path.read_bytes()),
                             config["data"]["validation_sha256"])
            train = [row for row in load_rows(train_path)
                     if row["split"] in config["data"]["train_splits"] and
                     row["source"] in config["data"]["train_sources"]]
            validation = [row for row in load_rows(validation_path)
                          if row["split"] in config["data"]["validation_splits"] and
                          row["source"] in config["data"]["validation_sources"]]
            self.assertEqual(len(train), config["data"]["train_rows"])
            self.assertEqual(len(validation), config["data"]["validation_rows"])
            expected_updates = (config["training"]["epochs"] *
                                math.ceil(len(train) / config["training"]["batch_size"]))
            self.assertEqual(expected_updates, config["training"]["updates"])
            cache = ROOT / config["feature_cache"]["path"] / "manifest.json"
            self.assertEqual(digest(cache.read_bytes()),
                             config["feature_cache"]["manifest_sha256"])
            for path, expected in config["source_files"].items():
                self.assertEqual(digest((ROOT / path).read_bytes()), expected)
            self.assertNotIn(config["evidence_path"], evidence_paths)
            self.assertNotIn(config["report_path"], report_paths)
            evidence_paths.add(config["evidence_path"])
            report_paths.add(config["report_path"])


if __name__ == "__main__":
    unittest.main()
