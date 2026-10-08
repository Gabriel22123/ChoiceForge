import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


path = Path(__file__).parents[1] / "scripts/verify_release_bundle.py"
spec = importlib.util.spec_from_file_location("verify_release_bundle", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseBundleVerifierTest(unittest.TestCase):
    def test_required_contract_names_adapter_and_public_evidence(self):
        self.assertIn("checkpoint/decision.safetensors", module.REQUIRED)
        self.assertIn("docs/evidence/release-candidate-v1.json", module.REQUIRED)

    def test_forbidden_private_paths_are_rejected(self):
        for name in ("blind-results/x.json", "x.private.json", "cache/x", "../escape"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.validate_release_path(name)

    def test_minimal_integrity_complete_bundle_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = {}
            for name in module.REQUIRED:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if name == "docs/evidence/release-candidate-v1.json":
                    data = json.dumps({"eligible": True,
                                       "release_endpoint": "seed1701-expanded"}).encode()
                else:
                    data = ("test:" + name).encode()
                target.write_bytes(data)
                records[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            (root / "release-manifest.json").write_text(json.dumps({
                "format_version": 1, "release_endpoint": "seed1701-expanded",
                "base_model_included": False, "canary_cases_included": False,
                "github_published": False, "files": records,
            }))
            result = module.verify_bundle(root, require_completion=False)
            self.assertTrue(result["verified"])
            self.assertEqual(result["files"], len(module.REQUIRED))

    def test_final_verification_requires_successful_completion_audit(self):
        self.assertIn("docs/evidence/project-completion-v1.json", module.COMPLETION_REQUIRED)


if __name__ == "__main__":
    unittest.main()
