# ChoiceForge project positioning

ChoiceForge has two connected outputs:

1. a knowledge-intensive decision study comparing decoder and encoder engines;
2. a reproducible audit protocol for candidate-decision systems.

It is not positioned as a general-purpose router. Routing and expert-bank
experiments remain in the repository because they document method boundaries
and negative transfer. They are historical evidence and reusable implementation
material, not the headline product claim.

## Research question

> How much world knowledge does a System 1 decision engine need, and which
> decision properties remain reliable when the task family changes?

The comparison should separate knowledge-intensive decisions from tasks that
can be solved by format or local rule learning. Whenever possible, decoder and
encoder runs should share the candidate interface, public-data budget,
calibration procedure and held-out task-family policy.

The primary measurements are hard-choice accuracy, NLL, Brier score, calibration
error, selective risk as coverage changes through abstention, sensitivity to
candidate order and context removal, task-family shift, latency and resource
cost.

Accuracy alone is insufficient. A model that preserves accuracy while making
probabilities more reliable can still help abstention and downstream audit. A
model that improves one task family but becomes overconfident under shift should
not advance.

## Audit protocol

Every published result should identify the data revision and license, selected
IDs, input transformation, model revision, configuration, random seed,
environment and output hashes. Development data may be inspected during
iteration; a sealed evaluation family is selected and hashed before labels or
final outcomes are opened.

The protocol reports positive and negative findings. It does not require the
reference model to win. Other candidate-decision engines can be evaluated when
they emit a candidate distribution and declare how abstention is handled.

## Evidence boundary

The released MiniCPM decoder bundle demonstrates stable probability-quality
improvement on one untouched public family under a frozen protocol. The
EuroBERT comparison is an architecture study, not a universal ranking of
decoder and encoder models. The current release does not establish universal
zero-shot competence or production readiness.

## Planned extensions

- a versioned benchmark manifest and one command that produces the audit record;
- knowledge-stratified and executable-outcome task families;
- first-class coverage-risk and calibration reports;
- repeated decoder/encoder comparisons at additional public model scales;
- explicit historical or ablation tracks for routing and expert-bank work.
