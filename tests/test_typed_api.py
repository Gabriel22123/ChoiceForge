import json
import unittest
from importlib.resources import files

from jsonschema import Draft202012Validator

from decision_model.typed_api import (format_typed_prediction, predict_typed,
                                      typed_to_request, validate_typed_prediction)


def base(choice_id, probabilities):
    return {
        "choice_id": choice_id,
        "probabilities": probabilities,
        "requires_review": True,
        "score_kind": "candidate_probability_not_a_guarantee",
        "calibration": "uncalibrated",
    }


class TypedApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema = json.loads(files("decision_model").joinpath(
            "schemas/typed-prediction.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        cls.validator = Draft202012Validator(schema)

    def test_boolean_has_exact_probability_semantics(self):
        payload = {"type": "boolean", "task": "Are all tests passing?",
                   "context": "23 passed, 1 failed.",
                   "criteria": {"true": "The full suite has no failures.",
                                "false": "At least one test failed."}}
        request = typed_to_request(payload)
        self.assertEqual([choice["id"] for choice in request["choices"]], ["true", "false"])
        result = format_typed_prediction(payload, base("false", {"true": .08, "false": .92}))
        self.assertEqual(result["value"], False)
        self.assertEqual(result["probability_true"], .08)
        self.assertAlmostEqual(result["margin"], .84)
        self.validator.validate(result)

    def test_choice_retains_dynamic_ids_and_top_two_margin(self):
        payload = {"type": "choice", "task": "Choose the next action.", "context": "Failure.",
                   "choices": [{"id": "inspect", "description": "Read the failed assertion."},
                               {"id": "retry", "description": "Rerun without changes."},
                               {"id": "submit", "description": "Submit the broken patch."}]}
        result = format_typed_prediction(
            payload, base("inspect", {"inspect": .7, "retry": .2, "submit": .1}))
        self.assertEqual(result["value"], "inspect")
        self.assertAlmostEqual(result["margin"], .5)
        self.validator.validate(result)

    def test_score_returns_argmax_level_and_expected_index(self):
        payload = {"type": "score", "task": "Rate deployment risk.", "context": "Auth changed.",
                   "levels": [{"id": "low", "description": "No external behavior changes."},
                              {"id": "medium", "description": "A public contract changed."},
                              {"id": "high", "description": "Authentication may be bypassed."}]}
        result = format_typed_prediction(
            payload, base("high", {"low": .1, "medium": .3, "high": .6}))
        self.assertEqual(result["level_index"], 2)
        self.assertAlmostEqual(result["expected_score"], 1.5)
        self.validator.validate(result)

    def test_rejects_free_form_output_and_inconsistent_derived_values(self):
        payload = {"type": "choice", "task": "Choose.", "context": "Evidence.",
                   "choices": [{"id": "a", "description": "A"},
                               {"id": "b", "description": "B"}]}
        result = format_typed_prediction(payload, base("a", {"a": .6, "b": .4}))
        result["reasoning"] = "unbounded"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_typed_prediction(result, payload)
        result.pop("reasoning")
        result["margin"] = .9
        with self.assertRaisesRegex(ValueError, "top-two"):
            validate_typed_prediction(result, payload)

    def test_predict_typed_uses_generic_judge(self):
        payload = {"type": "boolean", "task": "Is it safe?", "context": "Read-only command.",
                   "criteria": {"true": "No state changes.", "false": "State may change."}}

        class Judge:
            def predict(self, request):
                self.request = request
                return base("true", {"true": .75, "false": .25})

        judge = Judge()
        result = predict_typed(judge, payload)
        self.assertEqual(result["value"], True)
        self.assertEqual(judge.request, typed_to_request(payload))

    def test_strict_typed_inputs(self):
        bad = {"type": "boolean", "task": "Question", "context": "Context",
               "criteria": {"true": "Yes", "false": "No"}, "reasoning": "inject"}
        with self.assertRaisesRegex(ValueError, "exactly"):
            typed_to_request(bad)
        bad = {"type": "score", "task": "Score", "context": "Context",
               "levels": [{"id": "only", "description": "Only"}]}
        with self.assertRaisesRegex(ValueError, "2..10"):
            typed_to_request(bad)


if __name__ == "__main__":
    unittest.main()
