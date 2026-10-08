import copy
import json
import tempfile
import unittest
from pathlib import Path

from decision_model.core import (canonical, distribution_metrics, load_rows, metrics,
                                 render, target_distribution, validate_request)


def request():
    return {"task":"Choose a tool.","context":"Schedule a reminder.","choices":[
        {"id":"id_a","description":"Schedule events"},
        {"id":"id_b","description":"Search documents"}]}


class ContractTests(unittest.TestCase):
    def test_dynamic_choices_and_ids_not_encoded(self):
        a=request(); b=copy.deepcopy(a)
        b["choices"][0]["id"]="arbitrary_customer_id"
        self.assertEqual(render(a),render(b))
        for n in (2,3,7,32):
            a["choices"]=[{"id":str(i),"description":f"Candidate meaning {i}"} for i in range(n)]
            self.assertEqual(render(a).count("<|mask|>"),n)

    def test_no_duplicate_or_reserved_choices(self):
        a=request();a["choices"][1]["id"]=a["choices"][0]["id"]
        with self.assertRaises(ValueError):validate_request(a)
        a=request();a["context"]="inject <|mask|>"
        with self.assertRaises(ValueError):render(a)

    def test_mixed_choice_count_metrics(self):
        m=metrics([0,2],[[.7,.3],[.1,.2,.7]])
        self.assertEqual(m["correct"],2)
        self.assertAlmostEqual(m["uniform_random_accuracy"],(1/2+1/3)/2)
        self.assertIsNone(m["selective"]["0.9"]["accuracy"])

    def test_group_leakage_rejected(self):
        row={"id":"a","request":request(),"label":"id_a","split":"train","group_id":"family","provenance":[{"url":"public"}]}
        other=copy.deepcopy(row);other.update(id="b",split="test");other["request"]["context"]="another request"
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"rows.jsonl";p.write_text(canonical(row)+"\n"+canonical(other))
            with self.assertRaisesRegex(ValueError,"leakage"):load_rows(p)

    def test_reordered_duplicate_rejected(self):
        row={"id":"a","request":request(),"label":"id_a","split":"train","group_id":"family","provenance":[{"url":"public"}]}
        other=copy.deepcopy(row);other["id"]="b";other["request"]["choices"].reverse()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"rows.jsonl";p.write_text(canonical(row)+"\n"+canonical(other))
            with self.assertRaisesRegex(ValueError,"Duplicate model input"):load_rows(p)

    def test_model_input_contains_no_gold_metadata(self):
        self.assertNotIn("id_a",render(request()))
        self.assertEqual(set(validate_request(request())),{"task","context","choices"})

    def test_renamed_ids_do_not_hide_duplicate_inputs(self):
        row={"id":"a","request":request(),"label":"id_a","split":"train","group_id":"family","provenance":[{"url":"public"}]}
        other=copy.deepcopy(row);other["id"]="b"
        other["request"]["choices"][0]["id"]="renamed";other["label"]="renamed"
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"rows.jsonl";p.write_text(canonical(row)+"\n"+canonical(other))
            with self.assertRaisesRegex(ValueError,"Duplicate model input"):load_rows(p)

    def test_soft_target_is_validated_and_follows_candidate_permutation(self):
        row={"id":"a","request":request(),"label":"id_b","split":"train","group_id":"family",
             "target_probabilities":{"id_a":.25,"id_b":.75},"provenance":[{"url":"public"}]}
        self.assertEqual(target_distribution(row), [.25, .75])
        self.assertEqual(target_distribution(row, list(reversed(row["request"]["choices"]))), [.75, .25])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"rows.jsonl";path.write_text(canonical(row)+"\n")
            self.assertEqual(load_rows(path)[0]["label"], "id_b")
        for bad in ({"id_a":.5}, {"id_a":.6,"id_b":.6}, {"id_a":.75,"id_b":.25}):
            changed=copy.deepcopy(row);changed["target_probabilities"]=bad
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/"rows.jsonl";path.write_text(canonical(changed)+"\n")
                with self.assertRaises(ValueError):load_rows(path)

    def test_distribution_metrics_reduce_to_hard_metrics_and_retain_soft_signal(self):
        hard=distribution_metrics([[1.,0.],[0.,0.,1.]], [[.7,.3],[.1,.2,.7]])
        legacy=metrics([0,2], [[.7,.3],[.1,.2,.7]])
        for key in ("accuracy","nll","brier","ece_10"):
            self.assertAlmostEqual(hard[key], legacy[key])
        soft=distribution_metrics([[.6,.4]], [[.7,.3]])
        self.assertEqual(soft["argmax_agreement"], 1)
        self.assertAlmostEqual(soft["soft_brier"], .02)
        self.assertAlmostEqual(soft["total_variation"], .1)


if __name__=="__main__":unittest.main()
