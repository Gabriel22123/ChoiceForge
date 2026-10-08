# Typed Decisions soft targets and held-out workflows

## Scope

Typed Decisions adds complete Boolean, Choice and Score probability targets to
the project. Its targets average samples from a public teacher endpoint. They
measure teacher agreement, not objectively verified action success.

The data, training and metric paths are complete. A matched 13-case real-model
pilot runs hard targets, soft targets, typed ordered RPS and an independent
leave-one-out RLCD-style estimator. It proves integration and records numerical
direction only; no model-quality gain or method advancement is claimed.

## Pinned source

- Dataset: `LocalLLaMA/typed-decisions`
- Revision: `ea9306458d6e9563628369a3d1e72e362fb381d2`
- License: Apache-2.0
- Combined train Parquet SHA-256:
  `46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5`
- Combined test Parquet SHA-256:
  `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`

Only state, question instructions, candidate semantics and teacher
probabilities are retained. Latent generation factors, label-agreement
diagnostics, rationales and all other upstream columns are excluded.

## Four leave-one-workflow-out folds

Each of agent tracing, customer service, invoice processing and security
incidents is held out once. Its 300 upstream train cases are discarded in that
fold, and its 100 official test cases are zero-shot evaluation only. Each of the
other three workflows contributes 240 train, 60 validation and 100 official
test cases. Five questions stay grouped per case.

| split | cases | questions |
|---|---:|---:|
| train | 720 | 3,600 |
| validation | 180 | 900 |
| test | 400 | 2,000 |

Every fold must be reported. Selecting a workflow after observing results is
forbidden by the generated manifest.

## Objective and metrics

`target_probabilities` is keyed by candidate ID and realigned after every
candidate permutation. Legacy rows without it retain the original hard-label
path. For target `p` and prediction `q`, training uses soft cross entropy plus
weighted soft Brier:

```text
-sum_k p_k log(q_k) + beta * sum_k (q_k-p_k)^2
```

Evaluation reports full-target cross entropy, Brier, total variation, target
entropy and teacher-argmax agreement. Temperature is selected using validation
soft cross entropy only. Soft-target ECE compares confidence with target mass
on the predicted class and must not be described as real-world error calibration.

The pilot report is
[`TYPED_DECISIONS_RLCD_PILOT.zh-CN.md`](TYPED_DECISIONS_RLCD_PILOT.zh-CN.md),
with machine evidence in
[`typed-decisions-rlcd-pilot-v1.json`](evidence/typed-decisions-rlcd-pilot-v1.json).
The next quality study requires frozen multi-seed, four-fold
hard-target/soft-target/typed-RPS/RLCD comparisons under matched bases, examples,
update budgets and permutations.
