import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_qasc_dual_function_screen", ROOT / "scripts/run_qasc_dual_function_screen.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QascDualFunctionScreenTest(unittest.TestCase):
    def test_attach_teacher_and_weights(self):
        rows = [
            {"id": "r", "source": "clinc", "split": "train", "request": {"choices": [{"id": "a"}, {"id": "b"}]}, "teacher_probabilities": {"a": .7, "b": .3}},
            {"id": "q", "source": "qasc", "split": "train", "request": {"choices": [{"id": "a"}, {"id": "b"}]}},
            {"id": "v", "source": "qasc", "split": "validation", "request": {"choices": [{"id": "a"}, {"id": "b"}]}},
        ]
        result = MODULE.attach_teacher_and_weights(rows, {"r": {"a": .4, "b": .6}}, 1.0, .5)
        replay, qasc, validation = result
        self.assertEqual(replay["teacher_null_probabilities"], {"a": .4, "b": .6})
        self.assertEqual(replay["training_weight"], 1.0)
        self.assertEqual(qasc["training_weight"], .5)
        self.assertNotIn("training_weight", validation)

    def test_attach_rejects_missing_or_misaligned_teacher(self):
        row = {"id": "r", "source": "clinc", "split": "train", "request": {"choices": [{"id": "a"}, {"id": "b"}]}, "teacher_probabilities": {"a": .7, "b": .3}}
        with self.assertRaises(ValueError):
            MODULE.attach_teacher_and_weights([row], {}, 1.0, .5)
        with self.assertRaises(ValueError):
            MODULE.attach_teacher_and_weights([row], {"r": {"a": 1.0}}, 1.0, .5)


if __name__ == "__main__":
    unittest.main()
