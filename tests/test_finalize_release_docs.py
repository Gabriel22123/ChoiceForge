import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "scripts/finalize_release_docs.py"
spec = importlib.util.spec_from_file_location("finalize_release_docs", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FinalizeReleaseDocsTest(unittest.TestCase):
    def test_replace_block_is_exact_and_idempotent(self):
        original = "before\n" + module.START + "\nold\n" + module.END + "\nafter\n"
        once = module.replace_block(original, "new")
        twice = module.replace_block(once, "new")
        self.assertEqual(once, twice)
        self.assertIn("\nnew\n", once)

    def test_failed_result_never_names_release_checkpoint(self):
        evidence = {"eligible": False, "checks": {"a": True, "b": False},
                    "seen": {"mean_accuracy": {"control": .4, "expanded": .5}},
                    "blind": {"mean_accuracy": {"control": .3, "expanded": .4}}}
        result = module.bodies(evidence, "a" * 64)
        self.assertIn("no release checkpoint", result["README.md"])
        self.assertIn("Final release checkpoint: none", result["MODEL_CARD.md"])

    def test_eligible_result_records_exact_weight_hash(self):
        evidence = {"eligible": True, "checks": {"a": True},
                    "seen": {"mean_accuracy": {"control": .4, "expanded": .5}},
                    "blind": {"mean_accuracy": {"control": .3, "expanded": .4}}}
        value = "b" * 64
        result = module.bodies(evidence, value)
        self.assertIn(value, result["README.md"])
        self.assertIn("seed1701-expanded", result["MODEL_CARD.md"])


if __name__ == "__main__":
    unittest.main()
