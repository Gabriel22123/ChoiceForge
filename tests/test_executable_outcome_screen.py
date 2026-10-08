import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

from decision_model.feature_screen import CandidateHead


def load_script():
    path = Path(__file__).parents[1] / "scripts/run_executable_outcome_screen.py"
    spec = importlib.util.spec_from_file_location("outcome_screen", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixture_row(index, split, rewards):
    choices = [{"id": f"c{candidate}", "description": f"program {candidate}"}
               for candidate in range(len(rewards))]
    outcomes = {}
    for choice, reward in zip(choices, rewards):
        passed = int(reward * 2)
        outcomes[choice["id"]] = {"passed": passed, "total": 2, "reward": reward,
                                  "categories": ["passed"] * passed + ["assertion"] * (2 - passed)}
    best = max(range(len(rewards)), key=rewards.__getitem__)
    return {"id": f"r{index}", "split": split, "label": choices[best]["id"],
            "request": {"task": "choose", "context": str(index), "choices": choices},
            "candidate_outcomes": outcomes}


class ExecutableOutcomeScreenTests(unittest.TestCase):
    def test_all_arms_share_initial_policy_and_limited_arms_share_log(self):
        screen = load_script()
        rows = [fixture_row(0, "train", [1., 0.]),
                fixture_row(1, "train", [.5, 1., 0.]),
                fixture_row(2, "validation", [1., .5])]
        features = {row["id"]: torch.arange(len(row["request"]["choices"]) * 4,
                                              dtype=torch.float32).reshape(-1, 4).half()
                    for row in rows}
        protocol = {"study": "fixture", "training": {
            "epochs": 2, "batch_size": 2, "logged_actions_per_row_per_epoch": 3,
            "behavior_exploration": .2, "policy_head_width": 3, "policy_head_dropout": 0.,
            "reward_head_width": 3, "reward_head_dropout": 0., "learning_rate": .01,
            "weight_decay": 0., "gradient_clip": 1., "reward_model_weight": 1.}}
        torch.manual_seed(9); initial = CandidateHead.build(4, 3, 0.)
        plan = screen.make_plan(rows, protocol, 42)
        logs = screen.make_behavior_logs(rows, features, initial, protocol, 42)
        self.assertEqual(len(logs), 4)
        self.assertEqual({len(record["actions"]) for record in logs}, {3})
        runs = {}
        with tempfile.TemporaryDirectory() as directory:
            for arm in screen.ARMS:
                runs[arm] = screen.train_arm(
                    Path(directory) / arm, arm, rows, features, initial, plan, "plan-hash",
                    logs, "log-hash", protocol, 42)
        self.assertEqual({run["initial_policy_head_sha256"] for run in runs.values()}.__len__(), 1)
        self.assertEqual(runs["R-ips-loo"]["behavior_log_sha256"],
                         runs["D-doubly-robust"]["behavior_log_sha256"])
        self.assertFalse(runs["R-ips-loo"]["complete_reward_table_visible_to_training"])
        self.assertFalse(runs["D-doubly-robust"]["complete_reward_table_visible_to_training"])
        self.assertTrue(runs["O-full-information"]["complete_reward_table_visible_to_training"])


if __name__ == "__main__":
    unittest.main()
