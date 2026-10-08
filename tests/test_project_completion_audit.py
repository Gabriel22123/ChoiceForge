import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).parents[1] / "scripts/audit_project_completion.py"
spec = importlib.util.spec_from_file_location("audit_project_completion", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProjectCompletionAuditTest(unittest.TestCase):
    def test_completion_audit_names_original_goal_dimensions(self):
        source = path.read_text()
        for key in ("public_redistributable_data", "strict_json_probabilities_review",
                    "rlcd_deconstruction_and_outcome_feedback", "encoder_decoder_comparison",
                    "multitask_stability", "candidate_prior_stability",
                    "cross_task_generalization", "single_verified_adapter_bundle"):
            self.assertIn(key, source)


if __name__ == "__main__":
    unittest.main()
