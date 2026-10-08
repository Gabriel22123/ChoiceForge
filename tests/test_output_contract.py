import unittest
import json
from importlib.resources import files

from jsonschema import Draft202012Validator

from decision_model.output_contract import validate_prediction


REQUEST = {
    "task": "Choose",
    "context": "Evidence",
    "choices": [
        {"id": "left", "description": "Left option"},
        {"id": "right", "description": "Right option"},
    ],
}


def prediction():
    return {
        "choice_id": "left",
        "probabilities": {"left": 0.6, "right": 0.4},
        "requires_review": True,
        "score_kind": "candidate_probability_not_a_guarantee",
        "calibration": "validation_temperature",
    }


class OutputContractTest(unittest.TestCase):
    def test_packaged_json_schema_accepts_exact_contract(self):
        schema = json.loads(files("decision_model").joinpath("schemas/prediction.schema.json").read_text())
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(prediction())

    def test_accepts_exact_contract(self):
        value = prediction()
        self.assertIs(validate_prediction(value, REQUEST), value)

    def test_rejects_extra_free_form_output(self):
        value = prediction()
        value["reasoning"] = "unbounded text"
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_prediction(value, REQUEST)

    def test_rejects_missing_or_unnormalized_probabilities(self):
        for probabilities in ({"left": 1.0}, {"left": 0.8, "right": 0.4}):
            value = prediction()
            value["probabilities"] = probabilities
            with self.assertRaises(ValueError):
                validate_prediction(value, REQUEST)

    def test_json_object_key_order_is_irrelevant(self):
        value = prediction()
        value["probabilities"] = {"right": 0.4, "left": 0.6}
        validate_prediction(value, REQUEST)

    def test_rejects_wrong_choice_and_review_bypass(self):
        value = prediction()
        value["choice_id"] = "right"
        with self.assertRaisesRegex(ValueError, "maximum"):
            validate_prediction(value, REQUEST)
        value = prediction()
        value["requires_review"] = False
        with self.assertRaisesRegex(ValueError, "human review"):
            validate_prediction(value, REQUEST)

    def test_ties_use_request_order(self):
        value = prediction()
        value["probabilities"] = {"left": 0.5, "right": 0.5}
        validate_prediction(value, REQUEST)
        value["choice_id"] = "right"
        with self.assertRaisesRegex(ValueError, "first maximum"):
            validate_prediction(value, REQUEST)


if __name__ == "__main__":
    unittest.main()
