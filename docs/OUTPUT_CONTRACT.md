# Fixed prediction contract

The model never generates JSON tokens. It produces one scalar score for each
caller-supplied candidate. Application code normalizes those scores and builds a
fixed object:

```json
{
  "choice_id": "approve",
  "probabilities": {"approve": 0.73, "reject": 0.27},
  "requires_review": true,
  "score_kind": "candidate_probability_not_a_guarantee",
  "calibration": "validation_temperature"
}
```

The machine-readable schema is packaged at
`decision_model/schemas/prediction.schema.json`. The runtime validator in
`decision_model.output_contract.validate_prediction` additionally enforces the
relationships JSON Schema cannot express:

- the object has exactly five public fields;
- probability keys exactly match the request's candidate IDs; JSON object order is irrelevant;
- every probability is finite and in `[0, 1]`, and the values sum to one;
- `choice_id` is the first maximum-probability candidate in request order;
- the research preview always returns `requires_review: true`;
- `score_kind` and `calibration` use registered values.

Candidate descriptions, task text and context are model inputs. Candidate IDs
are routing identifiers and do not enter the model. The output contains no
free-form rationale, chain of thought, tool call or model-authored field.

`probabilities` are estimates over the candidate set supplied in that request.
Adding, removing or rewriting candidates changes the decision problem. A high
value is not a correctness guarantee and does not authorize automatic action.

## Typed views

The same generic candidate model accepts three strict request views through the
normal `predict` command:

- `choice`: the original 2–32 dynamic choices;
- `boolean`: explicit `true` and `false` criteria;
- `score`: 2–10 ordered level IDs and descriptions.

Typed responses are program-built and validated against
`decision_model/schemas/typed-prediction.schema.json`. Every response retains
the complete probability map, `requires_review`, score semantics and calibration
metadata. It also adds bounded deterministic fields:

- every type: `top_probability` and the top-two `margin`;
- Boolean: a JSON boolean `value` and `probability_true`;
- Score: the argmax `level_index` and `expected_score`, calculated as the
  probability-weighted zero-based level index.

These fields are calculated by application code from the same validated
distribution. The model cannot emit a rationale or add an arbitrary JSON key.
Runtime cross-field checks verify that the margin, Boolean mass, selected level
and expected score agree with the probability map.
