import copy
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from qasc_evidence_data import (OPTION_IDS, qasc_case, qasc_request_and_label,
                                select_position_balanced, selection_key)
from qasc_evidence_screen_report import prediction_marginal, source_deltas
from run_qasc_evidence_screen import raw_distribution, token_admissible


def source_row(index=0, answer="A"):
    labels = list("ABCDEFGH")
    return {
        "id": f"row-{index}",
        "question": f"Which answer follows for example {index}?",
        "choices": {
            "label": labels,
            "text": [f"candidate {label} for {index}" for label in labels],
        },
        "answerKey": answer,
        "fact1": f"First evidence for {index}.",
        "fact2": f"Second evidence for {index}.",
        "combinedfact": f"Combined answer-bearing fact for {index}.",
    }


class QascEvidenceScreenTests(unittest.TestCase):
    def test_selection_key_is_answer_blind_and_source_letters_are_hidden(self):
        row = source_row()
        changed = copy.deepcopy(row)
        changed["answerKey"] = "H"
        self.assertEqual(selection_key(row), selection_key(changed))
        request, label = qasc_request_and_label(row)
        self.assertIn(label, OPTION_IDS)
        self.assertEqual([choice["id"] for choice in request["choices"]], list(OPTION_IDS))
        self.assertFalse(any(choice["description"] in set("ABCDEFGH")
                             for choice in request["choices"]))

    def test_conversion_keeps_two_facts_and_excludes_combined_fact(self):
        row = source_row(3, "D")
        case = qasc_case(row, "train", "a" * 64)
        self.assertEqual(case["source"], "qasc")
        self.assertEqual(case["decision_type"], "choice")
        self.assertIn(row["fact1"], case["request"]["context"])
        self.assertIn(row["fact2"], case["request"]["context"])
        self.assertNotIn(row["combinedfact"], case["request"]["context"])

    def test_position_balanced_selection_is_exact_and_deterministic(self):
        rows = [source_row(index, "ABCDEFGH"[index % 8]) for index in range(256)]
        first = select_position_balanced(rows, 2, lambda request: True)
        second = select_position_balanced(list(reversed(rows)), 2, lambda request: True)
        self.assertEqual([row["id"] for row in first], [row["id"] for row in second])
        counts = Counter(qasc_request_and_label(row)[1] for row in first)
        self.assertEqual(counts, Counter({identity: 2 for identity in OPTION_IDS}))

    def test_temperature_inversion_recovers_raw_distribution(self):
        raw = {"a": 0.2, "b": 0.3, "c": 0.5}
        temperature = 2.0
        scaled = {key: value ** (1 / temperature) for key, value in raw.items()}
        total = sum(scaled.values())
        recovered = raw_distribution(
            {key: value / total for key, value in scaled.items()}, temperature)
        for key in raw:
            self.assertAlmostEqual(recovered[key], raw[key], places=12)

    def test_token_filter_uses_every_candidate_prompt(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=True):
                return text.split()
        request, _ = qasc_request_and_label(source_row())
        self.assertTrue(token_admissible(Tokenizer(), request, 200))
        self.assertFalse(token_admissible(Tokenizer(), request, 2))

    def test_prediction_marginal_and_source_delta_validate_coverage(self):
        cases = [qasc_case(source_row(index, "ABCDEFGH"[index]), "validation", "b" * 64)
                 for index in range(8)]
        predictions = []
        for row in cases:
            probabilities = {choice["id"]: 0.01 for choice in row["request"]["choices"]}
            probabilities[row["label"]] = 0.93
            predictions.append({"id": row["id"], "label": row["label"],
                                "probabilities": probabilities})
        marginal = prediction_marginal(cases, predictions)
        self.assertEqual(marginal["rows"], 8)
        self.assertEqual(marginal["prediction_truth_total_variation"], 0)
        with self.assertRaisesRegex(ValueError, "coverage"):
            prediction_marginal(cases, predictions[:1])
        candidate = {"by_source": {"a": {"accuracy": .8}}}
        control = {"by_source": {"a": {"accuracy": .5}}}
        self.assertAlmostEqual(source_deltas(candidate, control)["a"], .3)


if __name__ == "__main__":
    unittest.main()
