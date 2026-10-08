# Public data and evaluation protocol

All ingestion is restricted to the pinned source manifest. Source bytes are
verified before processing. No downloaded Python examples are executed.
Every row includes a URL, Git revision or archive version, source-file SHA-256, license and label
derivation. File paths, annotations and labels do not enter the model input.

## JSON Schema

Source: JSON-Schema-Test-Suite, draft2020-12 root test files. Each retained
upstream boolean is independently checked with jsonschema 4.25.1. References,
alternative dialects, format and regular-expression cases are excluded in this
version to avoid network resolution and engine-dependent semantics. Oversized
cases are counted and excluded, never truncated.

An ordinary example contains one candidate instance. Derived evidence examples
contain two possible instances: all conforming => satisfied, all failing =>
violated, mixed => insufficient. The alternatives are explicitly stated to be
exhaustive; this is a finite ambiguity construction, not human uncertainty
annotation. Two-candidate examples exist for all three labels. Original test
descriptions often reveal the answer and are excluded from the model input.
Identical schemas and their variants share one split.

## Bandit

Only four public example files are used: TLS verification, explicit timeout,
subprocess shell selection and YAML loader selection. We extract imports and
individual calls through Python AST parsing, remove comments and apply a narrow
local policy oracle. These are **derived policy labels**, not a claim that the
entire application is secure or that every upstream Bandit finding is correct.
The explicit-timeout policy intentionally differs from library defaults.
Missing dynamic values are insufficient; unknown imports/helper aliases are not
blindly labeled safe. The source file's original comments are not label truth.
All variants of a (policy, called API) stay in one split.

## CLINC150

The public crowdsourced data uses its original train/validation/test split.
Exact casefolded utterance duplicates are removed with test priority. Each
utterance is converted into one candidate-ranking question containing its true
intent and 1, 3 or 7 deterministically sampled distractors. Label descriptions
replace underscores with spaces. This is **not** the original 150-way benchmark;
do not compare these numbers directly to CLINC150 leaderboard accuracy.
Wikipedia-augmented and out-of-scope training files are not used in this version.
Hard-negative mining and out-of-scope evaluation remain future work.

## GoEmotions — held out as an entire task

Only the upstream test TSV is ingested. Keep single-label examples whose label
is anger, fear, gratitude, joy, neutral, sadness or surprise, and provide all
seven candidates. Multi-label and other-emotion examples are excluded. Every
row is assigned `test` and `evaluation_regime: held_out_task`. Training checks
that no row from that task family occurs in training or validation.

This is a filtered diagnostic, not the original 28-label multilabel benchmark.
English Reddit text has sampling/annotation biases and may contain offensive
language. Public benchmark contamination in base-model pretraining cannot be
ruled out. The task is unseen in this fine-tuning recipe, not proven unseen by
the pretrained encoder.

## SNLI and the second public mixture

`scripts/prepare_snli.py` verifies the pinned official SNLI 1.0 archive, preserves
gold labels and official split roles, removes cross-split normalized-premise
groups, conflicting pairs and duplicates, then takes a deterministic
label-balanced subset. See [the preparation record](SNLI_PREPARATION.md).
It provides 3,072 train, 384 validation and 768 test examples, under CC BY-SA 4.0.
The model receives premise/hypothesis text with three self-contained candidate
descriptions. No votes, source IDs or gold labels are included in model text.

`scripts/compose_public_data.py` combines the first public mixture with this
separate SNLI preparation, verifying both input hashes from
`configs/public-multitask-v2-inputs.json`. It retains every split, group, source
license and provenance record; then checks global duplicate and group constraints.
The resulting second mixture has 30,640 rows. Composition itself adds no labels
and does not turn previously inspected diagnostics into a fresh blind test.

The real-model gradient study selects only 192 continuation-training examples:
48 SNLI, 48 CLINC, 48 Schema and 48 Bandit. The 144 old-source examples were
already seen by the warm-start checkpoint. All three arms use the same selection.
SNLI evaluation is seen-task evaluation once these examples are trained.

## WinoGrande — frozen task-family transfer evaluation

WinoGrande 1.1 is never added to training or validation. The preparation script
verifies the official archive hash, checks the duplicated embedded/sidecar dev
labels, and chooses 384 of 1,267 development cases by the SHA-256 of `qID`,
sentence and both options. The answer is excluded from selection. Each row becomes
a two-candidate blank-filling decision with `evaluation_regime` set to
`untouched_task_family`.

All nine model endpoints are recorded before inference. Raw accuracy is primary;
no WinoGrande temperature fit, early stopping or checkpoint selection is allowed.
The test is unseen by project fine-tuning, while possible public-base pretraining
contamination cannot be ruled out. The official repository states CC-BY without
a version, so release bundles retain code, source hashes and aggregate results
but exclude raw/transformed WinoGrande text. See
`configs/winogrande-blind-source.json`.

## PAWS-Wiki training and BoolQ frozen task-family evaluation

`scripts/run_task_expansion_study.py` reads pinned parquet conversions of
PAWS-Wiki Labeled (Final). It selects 384 examples per label from official train,
48 per label from validation and 96 per label from test using deterministic
input hashes. Each pair becomes a two-candidate paraphrase decision. Training
adds 768 PAWS rows to the same 256 old-task replay rows; validation and test keep
their official roles and retain the previous relation/old-task diagnostics.

BoolQ never enters training or calibration. From the official development split,
384 token-admissible rows are selected by `SHA-256(question, passage)`; `answer`
is excluded from the selection key. Each row becomes a passage-grounded yes/no
decision with self-contained candidate descriptions. Both PAWS training endpoints
are completed before any BoolQ inference. WinoGrande reuses the byte-identical
previously frozen 384 rows and remains training-excluded.

PAWS-Wiki's official data permission allows free use and requests acknowledgement;
BoolQ is CC BY-SA 3.0. Parquet revisions and file hashes appear in the frozen
protocol. Bundles exclude transformed source text and include aggregate evidence,
preparation/evaluation code, license snapshots and checksums.

## BIG-bench logical_deduction — frozen candidate-prior confirmation

The official BIG-bench repository is pinned at commit
`092b196c1f8f14a54bbc62f24759d43bde46dd3b` under Apache-2.0. The preparation
script verifies the repository license, task README and three JSON task files.
For the 3-, 5- and 7-object subtasks, it admits all rows under the pinned decoder
tokenizer, sorts by `SHA-256(subtask, input, ordered choice text)` without target
scores, and retains 96 per subtask. Exact fine-tuning request overlap is zero.

The 288 transformed cases remain sealed until the shared prior-correction strength
has been selected from seed42/43 validation NLL and written with a fixed hash.
They never enter training, temperature fitting or method selection. Possible base
pretraining contamination cannot be excluded. Release bundles include acquisition
and transformation code, source/protocol hashes and aggregate results, while raw
or transformed questions, labels, logits and per-case predictions stay local.

## BIG-bench — release-candidate task-family confirmation

Release Candidate v1 pins the same official repository revision and verifies the
license, README and task JSON for `causal_judgment`,
`formal_fallacies_syllogisms_negation` and `movie_recommendation`. The source
files are cached before training, but the selected 384-case suite remains
unmaterialized until all six endpoint weights and order/ID audits are locked.

Each task admits complete examples below the 768-token candidate-path limit,
sorts them by a SHA-256 key over task, input and ordered candidate text without
target-score values, and takes 128. Exact request overlap with release training
data must be zero. The suite is used once for final confirmation and is prohibited
from training, calibration, early stopping, endpoint selection and hyperparameter
selection. Its raw/transformed rows, labels, logits and per-case predictions are
local canary material and must not enter release bundles.

## General data contract

Each JSONL row has:

```json
{
  "id": "unique-example-id",
  "group_id": "shared-origin-family",
  "source": "my-public-dataset",
  "family": "intent_routing",
  "split": "train",
  "evaluation_regime": "seen_task_new_examples",
  "request": {
    "task": "Choose the applicable action.",
    "context": "Please schedule a reminder.",
    "choices": [
      {"id": "schedule", "description": "Schedule an event."},
      {"id": "search", "description": "Search documents."}
    ]
  },
  "label": "schedule",
  "provenance": [{"url": "https://example.org/data", "license": "MIT"}]
}
```

Use `validation` and `test` for the other splits. A repeated example group may
not span splits. Duplicated inputs are rejected even if candidate order changes.
Changing only label IDs does not change model input. Custom data is the user's
responsibility; its presence does not make it publishable under the code license.

Rows may optionally include `target_probabilities`, keyed by every candidate ID.
They must be finite, normalized and agree with the required first-argmax `label`.
Candidate permutations realign the target by ID. Rows without this field retain
the original integer-label training path exactly; no confidence is invented.
The pinned Apache-2.0 Typed Decisions preparation supplies public teacher
distributions and uses four leave-one-workflow-out folds. See
`TYPED_DECISIONS_STUDY.zh-CN.md`; teacher agreement is not action correctness.

Rows derived from executable tasks may instead include `candidate_outcomes`,
keyed by every candidate ID. Each value contains exact `passed`, `total`,
`reward=passed/total` and one bounded result category per test. This is a reward
table, not a normalized class distribution. The `label` is the first candidate
with maximum reward only for compatibility and hard-label controls. Outcome-aware
training must explicitly choose complete-information or logged-action access;
the generic hard-label trainer does not silently reinterpret these rewards.

The first outcome source is pinned MBPP. Development executes only official
train/validation IDs and leaves official test IDs untouched until the protocol
is locked. Candidate selection and order are outcome-independent. See
`configs/mbpp-outcomes-source.json` and
`EXECUTABLE_OUTCOME_RLCD_DESIGN.zh-CN.md`.

## Scale and limitations

The first public builder produces 26,416 rows for the pinned source revisions:
22,495 CLINC ranking questions, 2,342 test-only emotion examples, 1,457 schema
examples and 122 code-policy examples. The pilot samples a much smaller subset,
balancing sources before families/labels. Per-run selected IDs and excluded
token-budget IDs are recorded. Rule transformations have not been independently
human reviewed. This dataset is a reproducible starting point, not a complete
training corpus for universal decision making.
