import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from decision_model.core import digest
from scripts.task_gradient_trust_report import _validate_trust, compare_metrics, validate_live_source


def step(alpha, raw, full, final, active=None):
    return {
        "combined_diagnostic_norm": final,
        "transformation": {"trust_region": {
            "alpha": alpha,
            "active": alpha < 1 - 1e-12 if active is None else active,
            "reference_raw_combined_norm": raw,
            "full_projected_combined_norm": full,
            "final_combined_norm": final,
            "bound_norm": raw,
        }},
    }


class TrustReportValidationTest(unittest.TestCase):
    def test_complete_metric_tree_includes_selective_outputs(self):
        calculated = {"n": 2, "accuracy": 0.5, "selective": {"0.7": {
            "coverage": 0.5, "accuracy": 1.0, "selected_count": 1}}}
        compare_metrics(calculated, calculated)
        changed = {"n": 2, "accuracy": 0.5, "selective": {"0.7": {
            "coverage": 0.5, "accuracy": 0.0, "selected_count": 1}}}
        with self.assertRaisesRegex(ValueError, "selective.0.7.accuracy"):
            compare_metrics(calculated, changed)

    def test_accepts_full_zero_and_partial_steps(self):
        self.assertLess(_validate_trust(step(1.0, 10.0, 8.0, 8.0)), 0)
        self.assertEqual(_validate_trust(step(0.0, 10.0, 12.0, 10.0)), 0)
        self.assertAlmostEqual(_validate_trust(step(0.4, 10.0, 12.0, 10.0)), 0)

    def test_rejects_bound_violation(self):
        with self.assertRaisesRegex(ValueError, "violated"):
            _validate_trust(step(0.4, 10.0, 12.0, 10.1))

    def test_rejects_inconsistent_active_flag(self):
        with self.assertRaisesRegex(ValueError, "active flag"):
            _validate_trust(step(0.4, 10.0, 12.0, 10.0, active=False))

    def test_rejects_unnecessary_restriction(self):
        with self.assertRaisesRegex(ValueError, "non-amplifying"):
            _validate_trust(step(0.5, 10.0, 9.0, 10.0))

    def test_frozen_source_allows_live_project_to_evolve(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            (root / "src").mkdir()
            (study / "frozen-source/src").mkdir(parents=True)
            (root / "src/example.py").write_text("new implementation\n")
            frozen = b"study implementation\n"
            (study / "frozen-source/src/example.py").write_bytes(frozen)
            validate_live_source(root, study, "src/example.py", digest(frozen))

    def test_rejects_changed_live_and_snapshot_sources(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            study = root / "study"
            (root / "src").mkdir()
            (study / "frozen-source/src").mkdir(parents=True)
            (root / "src/example.py").write_text("new\n")
            (study / "frozen-source/src/example.py").write_text("also new\n")
            with self.assertRaisesRegex(ValueError, "Frozen study source changed"):
                validate_live_source(root, study, "src/example.py", digest(b"old\n"))


if __name__ == "__main__":
    unittest.main()
