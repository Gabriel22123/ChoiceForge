import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "scripts/audit_release_data.py"
spec = importlib.util.spec_from_file_location("audit_release_data", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseDataAuditTest(unittest.TestCase):
    def test_exact_release_data_passes_public_provenance_audit(self):
        result = module.audit(Path(__file__).parents[1])
        self.assertTrue(result["passed"])
        self.assertEqual(result["arms"]["expanded"]["rows"], 5056)
        self.assertFalse(result["blind_suite_rows_included"])
        self.assertFalse(result["combined_dataset_relicensed_as_apache"])


if __name__ == "__main__":
    unittest.main()
