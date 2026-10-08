import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_qasc_retention_path", ROOT / "scripts/run_qasc_retention_path.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class QascRetentionPathTest(unittest.TestCase):
    def test_blend_states(self):
        parent = {"a": torch.tensor([0.0, 2.0]), "b": torch.tensor([[4.0]])}
        endpoint = {"a": torch.tensor([4.0, 6.0]), "b": torch.tensor([[0.0]])}
        result = MODULE.blend_states(parent, endpoint, 0.25)
        self.assertTrue(torch.equal(result["a"], torch.tensor([1.0, 3.0])))
        self.assertTrue(torch.equal(result["b"], torch.tensor([[3.0]])))

    def test_blend_rejects_bad_grid_or_state(self):
        state = {"a": torch.tensor([1.0])}
        with self.assertRaises(ValueError):
            MODULE.blend_states(state, state, 0.0)
        with self.assertRaises(ValueError):
            MODULE.blend_states(state, {"b": torch.tensor([1.0])}, 0.5)


if __name__ == "__main__":
    unittest.main()
