import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from counterfactual_evidence_report import source_deltas
from synthetic_evidence_data import generated_cases, pair_metrics

from decision_model.core import load_rows


class CounterfactualEvidenceScreenTests(unittest.TestCase):
    def test_groups_fix_claim_task_and_candidates_and_change_only_evidence(self):
        rows = generated_cases(4, 2)
        self.assertEqual(len(rows), 12)
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
        self.assertEqual(len(groups), 6)
        for selected in groups.values():
            self.assertEqual({row["label"] for row in selected}, {"no", "yes"})
            requests = [row["request"] for row in selected]
            self.assertEqual(requests[0]["task"], requests[1]["task"])
            self.assertEqual(requests[0]["choices"], requests[1]["choices"])
            contexts = [json.loads(request["context"]) for request in requests]
            self.assertEqual(contexts[0]["claim"], contexts[1]["claim"])
            self.assertNotEqual(contexts[0]["evidence"], contexts[1]["evidence"])

    def test_generated_rows_satisfy_dataset_contract_and_split_groups(self):
        path = Path(self.id().replace(".", "-") + ".jsonl")
        try:
            rows = generated_cases(3, 2)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            loaded = load_rows(path)
            self.assertEqual(len(loaded), 10)
            train_subjects = {
                json.loads(row["request"]["context"])["claim"].split(" is ")[0]
                for row in loaded if row["split"] == "train"
            }
            validation_subjects = {
                json.loads(row["request"]["context"])["claim"].split(" is ")[0]
                for row in loaded if row["split"] == "validation"
            }
            self.assertTrue(train_subjects.isdisjoint(validation_subjects))
        finally:
            path.unlink(missing_ok=True)

    def test_pair_metrics_requires_both_rows_and_counts_complete_groups(self):
        rows = generated_cases(1, 1)
        validation = [row for row in rows if row["split"] == "validation"]
        predictions = [{
            "id": row["id"], "label": row["label"],
            "probabilities": ({"no": .9, "yes": .1} if row["label"] == "no"
                              else {"no": .1, "yes": .9}),
        } for row in validation]
        result = pair_metrics(validation, predictions)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["complete_groups"], 1)
        with self.assertRaisesRegex(ValueError, "coverage"):
            pair_metrics(validation, predictions[:1])

    def test_source_delta_requires_same_sources(self):
        candidate = {"by_source": {"a": {"accuracy": .8}}}
        control = {"by_source": {"a": {"accuracy": .5}}}
        self.assertAlmostEqual(source_deltas(candidate, control)["a"], .3)
        with self.assertRaisesRegex(ValueError, "source sets"):
            source_deltas(candidate, {"by_source": {"b": {"accuracy": .5}}})


if __name__ == "__main__":
    unittest.main()
