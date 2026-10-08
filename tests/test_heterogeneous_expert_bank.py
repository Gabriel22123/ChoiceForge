import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "heterogeneous_expert_bank", ROOT / "scripts/run_heterogeneous_expert_bank.py")
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class HeterogeneousExpertBankTest(unittest.TestCase):
    def setUp(self):
        self.default = {"residual_width": 256, "router_width": 128, "dropout": 0.0,
                        "initial_gate_probability": 0.25}

    def test_endpoint_can_override_residual_width(self):
        model = MODULE.endpoint_model(self.default, {"model": {"residual_width": 512}}, 2048)
        self.assertEqual(model["residual_width"], 512)
        self.assertEqual(model["router_width"], 128)
        self.assertEqual(model["hidden_size"], 2048)

    def test_unknown_model_key_is_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.endpoint_model(self.default, {"model": {"other": 1}}, 2048)


if __name__ == "__main__":
    unittest.main()
