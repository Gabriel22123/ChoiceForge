import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / "scripts/run_heterogeneous_semantic_consensus_route.py"
SPEC = importlib.util.spec_from_file_location("heterogeneous_consensus_test", PATH)
MODULE = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MODULE)


class HeterogeneousConsensusTest(unittest.TestCase):
    def test_endpoint_model_keeps_expert_specific_width(self):
        default = {"residual_width": 256, "router_width": 128, "dropout": 0.0,
                   "initial_gate_probability": 0.25}
        model = MODULE.endpoint_model(default, {"model": {"residual_width": 512}}, 2048)
        self.assertEqual(512, model["residual_width"])
        self.assertEqual(2048, model["hidden_size"])

    def test_aggregate_uses_row_weighted_safety_counts(self):
        def run(source, rows, selected, unsafe, gain):
            return {"heldout_source": source, "evaluation": {
                "rows": rows, "selected_experts": selected, "unsafe_selected_rows": unsafe,
                "path_match": 0.5, "mean_proper_gain": gain, "mean_regret": 0.1,
                "accuracy_delta": 0.0, "cross_entropy_delta": 0.0,
                "expert_selection_rate": selected / rows,
                "safe_selection_precision": 1 - unsafe / selected,
                "violation_count": unsafe,
            }}
        aggregate, by_source = MODULE.aggregate_results(
            [run("a", 10, 2, 1, 0.1), run("b", 20, 8, 1, 0.2)], ["a", "b"])
        self.assertEqual({"a", "b"}, set(by_source))
        self.assertAlmostEqual(0.8, aggregate["safe_selection_precision"])
        self.assertAlmostEqual(10 / 30, aggregate["expert_selection_rate"])


if __name__ == "__main__":
    unittest.main()
