import unittest
import torch
from decision_model.objective import proper_loss, consistency_loss
from decision_model.train import (balanced_subset, calibrate, evaluate_rows,
                                  training_distribution_field, training_targets)


class ObjectiveTests(unittest.TestCase):
    def test_strictly_proper_identity(self):
        p=torch.tensor([[.2,.3,.5]])
        q=torch.tensor([[.4,.4,.2]])
        delta=proper_loss(q.log(),p)-proper_loss(p.log(),p)
        expected=(p*(p.log()-q.log())).sum()+.5*((p-q)**2).sum()
        self.assertAlmostEqual(delta.item(),expected.item(),places=6)

    def test_gradient_at_true_distribution(self):
        p=torch.tensor([[.15,.35,.5]])
        logits=p.log().requires_grad_();proper_loss(logits,p).backward()
        self.assertLess(logits.grad.abs().max().item(),1e-6)

    def test_candidate_alignment(self):
        a=torch.tensor([[2.,0.,-1.]],requires_grad=True)
        b=a[:,[2,0,1]]
        self.assertGreater(consistency_loss(a,b).item(),0)
        self.assertAlmostEqual(consistency_loss(a,b[:,[1,2,0]]).item(),0,places=6)

    def test_padded_choices_do_not_change_loss(self):
        a=torch.tensor([[1.,0.]])
        y=torch.tensor([0])
        self.assertAlmostEqual(proper_loss(a,y).item(),proper_loss(torch.tensor([[1.,0.,-1e4]]),y).item(),places=6)

    def test_calibration_identity_is_candidate(self):
        rows=[{"request":{"choices":[{"id":"x"},{"id":"y"}]},"label":"x"}]*2
        result=calibrate([[1.,0.],[0.,1.]],rows)
        self.assertLessEqual(result["validation_nll"],result["identity_validation_nll"]+1e-10)

    def test_soft_calibration_and_evaluation_use_full_distribution(self):
        rows=[{"id":"a","source":"typed","family":"workflow","evaluation_regime":"held_out_task",
               "decision_type":"boolean","label":"y","target_probabilities":{"x":.2,"y":.8},
               "request":{"choices":[{"id":"x"},{"id":"y"}]}}]
        result=calibrate([[0.,1.]],rows)
        self.assertIn("soft cross entropy",result["method"])
        evaluated,predictions=evaluate_rows(rows,[[0.,1.]])
        self.assertIn("total_variation",evaluated)
        self.assertEqual(evaluated["by_family"]["workflow"]["n"],1)
        self.assertEqual(evaluated["by_evaluation_regime"]["held_out_task"]["n"],1)
        self.assertEqual(evaluated["by_decision_type"]["boolean"]["n"],1)
        self.assertEqual(predictions[0]["target_probabilities"],{"x":.2,"y":.8})

    def test_training_targets_preserve_hard_path_and_align_soft_ids(self):
        hard={"label":"a","request":{"choices":[{"id":"a"},{"id":"b"}]}}
        hard_target=training_targets([hard],[{"choices":[{"id":"b"},{"id":"a"}]}],2,"cpu")
        self.assertEqual(hard_target.dtype,torch.int64)
        self.assertEqual(hard_target.tolist(),[1])
        soft={"label":"a","target_probabilities":{"a":.7,"b":.3},"request":hard["request"]}
        soft_target=training_targets([soft],[{"choices":[{"id":"b"},{"id":"a"}]}],3,"cpu")
        self.assertTrue(torch.allclose(soft_target,torch.tensor([[.3,.7,0.]]),atol=0,rtol=0))
        forced_hard=training_targets([soft],[{"choices":[{"id":"b"},{"id":"a"}]}],2,"cpu","hard")
        self.assertEqual(forced_hard.tolist(),[1])
        with self.assertRaisesRegex(ValueError,"requires probabilities"):
            training_targets([hard],[hard["request"]],2,"cpu","distribution")

    def test_auxiliary_distribution_aligns_ids_and_rejects_bad_support(self):
        row={"teacher_probabilities":{"a":.7,"b":.3},
             "request":{"choices":[{"id":"a"},{"id":"b"}]}}
        target=training_distribution_field(
            [row],[{"choices":[{"id":"b"},{"id":"a"}]}],3,"cpu",
            "teacher_probabilities")
        self.assertTrue(torch.allclose(target,torch.tensor([[.3,.7,0.]]),atol=0,rtol=0))
        bad=dict(row,teacher_probabilities={"a":1.0})
        with self.assertRaisesRegex(ValueError,"Invalid auxiliary"):
            training_distribution_field(
                [bad],[row["request"]],2,"cpu","teacher_probabilities")
        target, active=training_distribution_field(
            [row,{"request":row["request"]}],
            [row["request"],row["request"]],2,"cpu","teacher_probabilities",True)
        self.assertEqual(active.tolist(),[1.0,0.0])
        self.assertEqual(target[1].tolist(),[0.0,0.0])

    def test_source_balancing(self):
        rows=[{"id":str(i),"source":"a" if i<90 else "b","family":"f","label":"x"} for i in range(100)]
        subset=balanced_subset(rows,10)
        self.assertEqual(sum(r["source"]=="a" for r in subset),5)


if __name__=="__main__":unittest.main()
