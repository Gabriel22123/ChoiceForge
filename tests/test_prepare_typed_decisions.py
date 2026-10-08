import json
import unittest

from decision_model.core import render, target_distribution
from scripts.prepare_typed_decisions import build_fold, convert_case, normalize_question


def make_case(workflow, index, prefix="tr"):
    questions = {
        "accept": {"type":"noul", "instructions":"Should this be accepted?",
                   "criteria":{"true":"Evidence supports acceptance", "false":"Evidence supports rejection"}},
        "route": {"type":"choice", "instructions":"Where should it go?",
                  "criteria":{"fast":"Fast lane", "manual":"Manual review"}},
        "severity": {"type":"score", "instructions":"How severe is it?",
                     "criteria":["Low", "Medium", "High"]},
    }
    gold = {
        "accept":{"probabilities":{"true":.7,"false":.3}},
        "route":{"probabilities":{"fast":.25,"manual":.75}},
        "severity":{"probabilities":{"0":.1,"1":.2,"2":.7}},
    }
    return {"id":f"{prefix}_{workflow}_{index}","workflow":workflow,
            "state":json.dumps({"ticket":index,"workflow":workflow}),
            "questions":json.dumps(questions),"gold":json.dumps(gold),
            "factors":"SECRET LATENT FACTOR","label_agreement":1.0}


class TypedDecisionsPreparationTests(unittest.TestCase):
    def test_conversion_retains_probabilities_and_excludes_latent_fields(self):
        case=make_case("alpha",0)
        rows=convert_case(case,"train","train","a"*40,"Apache-2.0","seen_task_training")
        self.assertEqual(len(rows),3)
        boolean=next(row for row in rows if row["id"].endswith(":accept"))
        self.assertEqual(boolean["label"],"true")
        self.assertEqual(boolean["decision_type"],"boolean")
        self.assertEqual(target_distribution(boolean),[.7,.3])
        text=render(boolean["request"])
        self.assertNotIn("SECRET",text)
        self.assertNotIn("label_agreement",json.dumps(boolean))

    def test_four_way_question_shapes(self):
        self.assertEqual(normalize_question({"type":"score","instructions":"Rank", "criteria":["L","H"]})["type"],"score")
        default_boolean=normalize_question({"type":"noul","instructions":"Decide"})
        self.assertEqual(default_boolean["choices"],[{"id":"true","description":"TRUE"},
                                                      {"id":"false","description":"FALSE"}])
        with self.assertRaises(ValueError):
            normalize_question({"type":"noul","instructions":"Decide", "criteria":{"true":"yes"}})

    def test_fold_excludes_heldout_train_and_keeps_all_official_test(self):
        workflows=["alpha","beta","gamma","delta"]
        train=[make_case(workflow,index) for workflow in workflows for index in range(3)]
        test=[make_case(workflow,0,"te") for workflow in workflows]
        config={"revision":"a"*40,"license":"Apache-2.0","split_seed":9,
                "expected":{"workflows":workflows,"train_cases_per_workflow":3,
                            "test_cases_per_workflow":1,"questions_per_case":3,
                            "seen_train_cases_per_workflow":2}}
        rows,counts=build_fold(train,test,"delta",config)
        self.assertEqual(counts["train"],{"alpha":2,"beta":2,"gamma":2})
        self.assertEqual(counts["validation"],{"alpha":1,"beta":1,"gamma":1})
        self.assertEqual(counts["test"],{workflow:1 for workflow in workflows})
        self.assertFalse(any(row["family"]=="delta" and row["split"]!="test" for row in rows))
        self.assertEqual({row["evaluation_regime"] for row in rows if row["family"]=="delta"},
                         {"held_out_task"})

    def test_gold_key_mismatch_is_rejected(self):
        case=make_case("alpha",0)
        gold=json.loads(case["gold"]);del gold["route"]["probabilities"]["fast"]
        case["gold"]=json.dumps(gold)
        with self.assertRaisesRegex(ValueError,"differ"):
            convert_case(case,"train","train","a"*40,"Apache-2.0","seen_task_training")


if __name__=="__main__":unittest.main()
