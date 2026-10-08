# ChoiceForge publication review

This is the review boundary for the first public ChoiceForge source and model
release. The public repository is
https://github.com/Gabriel22123/ChoiceForge. The source tree and the verified
ChoiceForge parameter bundle are published; the third-party MiniCPM base is
downloaded separately from its upstream revision.

## What the project is

ChoiceForge accepts a task, context and 2–32 caller-supplied candidates. It
returns exactly five fields: the selected candidate ID, candidate probabilities,
a mandatory review marker, score semantics and calibration status. It does not
generate reasoning text or arbitrary JSON. Quality-rule assessment is one
example rather than a fixed label space.

The model uses the public `openbmb/MiniCPM5-2B-Base` revision
`96a57cd572a02506b4500f54427dca24970c1bac`. The base remains separate and is
verified file by file. The local endpoint contains a parent decision checkpoint
and one merged residual adapter trained only from attributed public sources.

## Current evidence

The development sequence retains positive and negative results rather than
selecting only successful runs. Its main contribution is a practical
deconstruction of RLCD-style learning:

1. use public outcome or probability supervision where it exists;
2. treat proper probability loss and context use as first-class objectives;
3. train source specialists when one global update causes interference;
4. learn a source-balanced convex expert portfolio;
5. at inference, take the largest portfolio step that preserves the parent's
   hard decision;
6. apply the same step to the full-context and context-removed paths.

The decision-stable portfolio first passed a 27-fold leave-one-family-out study
over consumed public tasks. Three full-source seeds were then frozen before the
SciFact archive was opened.

On 209 answer-blind-selected SciFact claim/evidence pairs, every seed passed all
pre-registered gates:

| Seed | Accuracy delta | Cross-entropy delta | Brier delta | Proper gain | Context delta |
|---:|---:|---:|---:|---:|---:|
| 42 | +0.0000 | -0.0730 | -0.0149 | +0.0805 | +0.0809 |
| 43 | +0.0000 | -0.0730 | -0.0149 | +0.0805 | +0.0809 |
| 44 | +0.0000 | -0.0739 | -0.0150 | +0.0814 | +0.0818 |

Strict output-contract and candidate-permutation checks pass on every row and
seed. The hard-choice guarantee makes the accuracy delta exactly zero; the new
evidence is a stable probability-quality and context-use improvement on an
untouched task family.

The seed-42 portfolio is canonical because 42 is the lowest registered seed, a
rule independent of observed results. Its six candidate-wise residual experts
are algebraically fused into one width-1792 adapter. Across 5,765 cached public
rows, full-context, context-removed and projected logits differ by at most
`1.43e-6`; every projection step and hard choice matches. The adapter SHA-256 is
`8e4c816e43ea70ccac43145646583e6466357455bc794761faecbe0fe23f196a`.

## Model package prepared locally

`models/choiceforge-local-bundle-v1` is the published 55,150,459-byte bundle:

- merged adapter: about 14 MB;
- parent decision checkpoint: about 39 MB;
- parent config, calibration and checksummed metadata;
- no base-model weights, optimizer state, training rows, feature cache or raw
  benchmark text.

The bundle manifest SHA-256 is
`27406c0544c8d44813a2e2363029ee265548997e42802f790eca0d3588fb59ba`.
The real command-line loading path was exercised on `examples/request.json` and
returned `calendar` with a normalized three-candidate distribution and
`requires_review: true`.

## Scientific limits

- SciFact tests classification given gold rationale sentences, not retrieval.
- One untouched family does not prove universal zero-shot competence.
- Public-base pretraining contamination cannot be excluded.
- Decision stability preserves hard choices; it does not guarantee every row's
  probability improves.
- The frozen endpoint is an exact fusion of the validated portfolio, not an
  independently retrained model.
- Every downstream use needs its own validation and human review.

## Published source contents

- Apache-2.0 project source, tests and continuous-integration workflow.
- Frozen protocols, aggregate evidence and reports, including retained failures.
- Public-source attribution, revisions, licenses and SHA-256 records.
- Reproducible data preparation, training, evaluation, fusion and bundle scripts.
- The 3,008-row Control and 5,056-row Expanded redistributable public-derived
  datasets already tracked with row-level provenance.
- The verified ChoiceForge parameter bundle and a loader for the separately
  downloaded public base.

## Excluded from the repository

- `runs/`, `models/`, `cache/`, `dist/`, virtual environments and downloaded
  archives.
- The 2B public base weights; users fetch the pinned upstream revision.
- Optimizer states, feature caches, raw or transformed restricted benchmark text,
  blind-suite questions, labels, logits and per-case predictions.
- Private data, private URLs, local absolute paths, credentials and environment
  files.

## Data and license boundary

Every fine-tuning and evaluation source is listed in `THIRD_PARTY.md`. The
SciFact claims/evidence annotations are CC BY 4.0, the abstract corpus is
ODC-By 1.0 and code is Apache 2.0. SciFact raw and transformed text remains in
ignored local paths; the release retains only source metadata, checksums,
protocol and aggregate evidence. The public base and project source are
Apache-2.0. Dataset licenses continue to apply to their derived rows.

## Publication checks

1. Full unit suite passes.
2. Repository scan finds no private markers, credentials, private URLs or local
   absolute paths in tracked content or history intended for publication.
3. No ignored weights, runs, caches, archives or environment files are staged.
4. Public-data provenance and license audit passes.
5. Git author is `Gabriel22123 <Gabriel22123@users.noreply.github.com>`.
6. The published repository contains only the reviewed source and parameter
   bundle; no company data, internal material, credentials or local paths are
   present.
7. The public README documents how to install, infer, train and verify the
   bundle.
