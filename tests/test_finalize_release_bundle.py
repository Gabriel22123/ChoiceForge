import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "scripts/finalize_release_bundle.py"
spec = importlib.util.spec_from_file_location("finalize_release_bundle", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FinalizeReleaseBundleTest(unittest.TestCase):
    def test_final_files_are_completion_report_and_evidence_only(self):
        self.assertEqual(set(module.FILES), {
            "docs/PROJECT_COMPLETION_AUDIT.zh-CN.md",
            "docs/evidence/project-completion-v1.json",
        })


if __name__ == "__main__":
    unittest.main()
