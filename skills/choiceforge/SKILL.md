---
name: choiceforge
description: Use the ChoiceForge public candidate-decision model for structured inference, public-data training, evaluation, or audit when the user wants a bounded choice with probabilities instead of free-form generation.
---

# ChoiceForge

Use this skill when a user wants to run or continue the public ChoiceForge model,
prepare a candidate-decision request, train an adapter, evaluate a checkpoint, or
explain the model's output contract. Keep the workflow task-agnostic: the model
accepts a task, optional context, and caller-supplied candidates.

## Choose a mode

- **Inference:** turn the user's task, context, and candidates into a canonical
  JSON request, run the verified bundle or a checkpoint, and report the returned
  choice, probabilities, calibration and review marker.
- **Training:** use only data with a public source, license, revision and
  provenance. Create a fresh output directory, preserve the config and seed, and
  never tune against a sealed test split.
- **Evaluation/audit:** run the requested split with a new output directory;
  separate measured metrics from claims about generalization.
- **Explanation:** describe candidate scoring, calibrated probabilities and the
  fixed JSON contract without presenting the model as an autonomous executor.

## Inference rules

1. Require at least two candidates with stable IDs and descriptions. Do not
   invent a candidate or silently discard one.
2. Keep the request in `task`, `context`, and `choices`; do not ask the model to
   generate a JSON object or a reasoning trace.
3. Prefer `predict-merged` with the verified bundle for normal use. Use
   `predict` only when the user explicitly selects an adapter checkpoint.
4. Present the runtime result as a decision aid. Preserve `requires_review: true`
   and do not convert probabilities into an irreversible action.

## Training and evaluation rules

- Start with the checked-in public release data and config unless the user names
  another public dataset.
- Before training, check the dataset manifest and provenance; reject missing or
  non-public sources instead of guessing their license.
- Use `--init-from` for a warm start, but write to a new output directory and
  record the parent checkpoint.
- Keep test data for the final report. Do not select a checkpoint, seed or
  hyperparameter from test metrics.
- After a run, use `evaluate` and `audit` as separate checks. Report failures and
  negative results rather than selecting a favorable subset.

Read [references/workflow.md](references/workflow.md) when the user asks for
concrete commands, input schemas, training continuation, or release checks.
