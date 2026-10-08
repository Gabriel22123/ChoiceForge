import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts/run_public_specialist.py"
SPEC = importlib.util.spec_from_file_location("run_public_specialist", SCRIPT)
ARC = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(ARC)


class ArcSpecialistTrainingTests(unittest.TestCase):

    def test_data_selection_can_hold_out_one_public_source(self):
        rows = [{"id": "a", "split": "train", "source": "x"},
                {"id": "b", "split": "validation", "source": "x"},
                {"id": "c", "split": "validation", "source": "y"}]
        self.assertEqual([row["id"] for row in
                          ARC.select_data_rows(rows, ["validation"], ["x"])], ["b"])
        with self.assertRaisesRegex(ValueError, "empty"):
            ARC.select_data_rows(rows, ["test"], ["x"])

    def test_plan_is_deterministic_and_permutation_complete(self):
        rows = [{"id": "b", "request": {"choices": [{}, {}, {}]}},
                {"id": "a", "request": {"choices": [{}, {}]}}]
        first = ARC.make_plan(rows, 42, 2)
        second = ARC.make_plan(list(reversed(rows)), 42, 2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        for record in first:
            count = next(len(row["request"]["choices"]) for row in rows
                         if row["id"] == record["row_id"])
            self.assertEqual(sorted(record["candidate_order"]), list(range(count)))

    def test_loss_rewards_correct_context_and_is_finite(self):
        parent = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        parent_null = torch.zeros_like(parent)
        mask = torch.ones_like(parent, dtype=torch.bool)
        targets = torch.tensor([0, 1])
        neutral = ARC.specialist_loss(parent, parent_null, torch.zeros_like(parent),
                                      torch.zeros_like(parent), mask, targets, .5, .25, .05)
        correction = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])
        improved = ARC.specialist_loss(parent, parent_null, correction,
                                       torch.zeros_like(parent), mask, targets, .5, .25, .05)
        self.assertTrue(torch.isfinite(neutral))
        self.assertLess(float(improved), float(neutral))


if __name__ == "__main__":
    unittest.main()
