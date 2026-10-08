import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "scripts/release_candidate_report.py"
spec = importlib.util.spec_from_file_location("release_candidate_report", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def metric(accuracy, loss, source_accuracy=None, soft=False):
    return {
        "accuracy": accuracy,
        ("cross_entropy" if soft else "nll"): loss,
        "by_source": {
            source: {"accuracy": value, "nll": loss}
            for source, value in (source_accuracy or {"task": accuracy}).items()
        },
    }


class ReleaseCandidateReportTest(unittest.TestCase):
    def test_probability_loss_maps_hard_and_soft_targets(self):
        self.assertEqual(module.probability_loss(metric(.5, .8)), .8)
        self.assertEqual(module.probability_loss(metric(.5, .7, soft=True)), .7)

    def test_paired_gate_uses_strict_wins_and_worst_task_seed(self):
        values = {}
        for seed in module.SEEDS:
            values[(seed, "control")] = metric(.60, .8, {"old": .60})
            values[(seed, "expanded")] = metric(.61 if seed != 1703 else .60, .7,
                                                 {"old": .59})
        result = module.paired_gate(values, ["old"])
        self.assertEqual(result["paired_expanded_accuracy_wins"], 2)
        self.assertAlmostEqual(result["minimum_task_seed_accuracy_delta"], -.01)
        self.assertGreater(result["retained_task_mean_accuracy"]["control"],
                           result["retained_task_mean_accuracy"]["expanded"])

    def test_probability_loss_rejects_ambiguous_record(self):
        with self.assertRaises(ValueError):
            module.probability_loss({"nll": 1, "cross_entropy": 1})


if __name__ == "__main__":
    unittest.main()
