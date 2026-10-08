import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "scripts/package_release_candidate.py"
spec = importlib.util.spec_from_file_location("package_release_candidate", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseBundleTest(unittest.TestCase):
    def test_forbidden_canary_tokens_are_explicit(self):
        source = path.read_text()
        for token in ("cases.jsonl", ".private.json", "blind-results", "cache/"):
            self.assertIn(token, source)

    def test_canonical_endpoint_is_predesignated_seed(self):
        self.assertEqual(module.CANONICAL, "seed1701-expanded")

    def test_packaged_schemas_use_the_installable_package_path(self):
        source = path.read_text()
        self.assertIn('root / "src/decision_model/schemas"', source)
        self.assertNotIn('root / "schemas"', source)

    def test_exact_release_data_and_retraining_entrypoints_are_allowlisted(self):
        self.assertEqual(set(module.RELEASE_DATA_FILES), {
            "data/release-candidate-v1/control.jsonl",
            "data/release-candidate-v1/expanded.jsonl",
            "data/release-candidate-v1/manifest.json",
        })
        self.assertIn("scripts/run_release_candidate_study.py", module.RELEASE_SCRIPT_FILES)
        self.assertIn("scripts/package_release_candidate.py", module.RELEASE_SCRIPT_FILES)


if __name__ == "__main__":
    unittest.main()
