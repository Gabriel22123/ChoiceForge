import importlib.util
import tempfile
import unittest
from pathlib import Path

from decision_model.core import canonical


def load_script():
    path = Path(__file__).parents[1] / "scripts/cache_typed_decisions_decoder_features.py"
    spec = importlib.util.spec_from_file_location("typed_feature_cache", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class TypedFeatureCacheTests(unittest.TestCase):
    def test_union_accepts_split_changes_but_rejects_semantic_changes(self):
        module = load_script()
        base = {"id": "x", "source": "typed-decisions", "family": "a", "group_id": "g",
                "decision_type": "boolean", "label": "yes",
                "provenance": {"source": "public-test-fixture"},
                "target_probabilities": {"yes": .7, "no": .3},
                "request": {"task": "q", "context": "c", "choices": [
                    {"id": "yes", "description": "yes"}, {"id": "no", "description": "no"}]}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, split in enumerate(("train", "test", "validation", "test")):
                fold = root / f"heldout-f{index}"; fold.mkdir()
                row = dict(base, split=split, evaluation_regime="held_out_task" if split == "test" else "seen")
                (fold / "cases.jsonl").write_text(canonical(row) + "\n")
            rows, paths = module.collect_unique_rows(root)
            self.assertEqual(len(paths), 4); self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][2], ["f0", "f1", "f2", "f3"])
            changed = dict(base, request=dict(base["request"], context="changed"),
                           split="test", evaluation_regime="held_out_task")
            (root / "heldout-f3/cases.jsonl").write_text(canonical(changed) + "\n")
            with self.assertRaisesRegex(ValueError, "different semantic"):
                module.collect_unique_rows(root)


if __name__ == "__main__":
    unittest.main()
