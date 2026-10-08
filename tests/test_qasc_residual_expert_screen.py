import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_qasc_residual_expert_screen",
              ROOT / "scripts/run_qasc_residual_expert_screen.py")
report = load("qasc_residual_expert_screen_report",
              ROOT / "scripts/qasc_residual_expert_screen_report.py")


class ResidualExpertScreenTest(unittest.TestCase):
    def test_parent_invariance_requires_subset_and_reports_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent.safetensors"
            endpoint = Path(directory) / "endpoint.safetensors"
            save_file({"head.weight": torch.tensor([[1.0, 2.0]])}, str(parent))
            save_file({"head.weight": torch.tensor([[1.0, 2.0]]),
                       "residual_head.0.weight": torch.tensor([3.0])}, str(endpoint))
            value = runner.parent_parameter_difference(parent, endpoint)
            self.assertTrue(value["parent_parameters_exact"])
            self.assertEqual(value["parent_max_absolute_difference"], 0.0)
            self.assertEqual(value["residual_tensors"], ["residual_head.0.weight"])

    def test_parent_invariance_detects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent.safetensors"
            endpoint = Path(directory) / "endpoint.safetensors"
            save_file({"head.weight": torch.tensor([[1.0, 2.0]])}, str(parent))
            save_file({"head.weight": torch.tensor([[1.0, 2.1]]),
                       "residual_head.0.weight": torch.tensor([3.0])}, str(endpoint))
            value = runner.parent_parameter_difference(parent, endpoint)
            self.assertFalse(value["parent_parameters_exact"])
            self.assertGreater(value["parent_max_absolute_difference"], 0.0)

    def test_advancement_requires_every_gate(self):
        self.assertTrue(report.advancement({"a": True, "b": True}))
        self.assertFalse(report.advancement({"a": True, "b": False}))


if __name__ == "__main__":
    unittest.main()
