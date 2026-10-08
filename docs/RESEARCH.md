# Research direction: calibrated zero-shot decisions

**Public RLCD ideas → explicit hypotheses → independent implementations → reproducible evidence.**

The project studies how a model can select among caller-defined candidates,
communicate useful probabilities, remain stable under equivalent formulations,
and transfer to task families absent from its fine-tuning. Quality gates are
one application. Universal zero-shot competence is the long-term objective,
not a current result.

See the [Chinese explanation](RESEARCH.zh-CN.md) for the motivation, mathematical
decomposition and distinction between decision contracts and semantic quality.

## What is public, and what is ours

The project uses publicly describable calibrated-decision objectives and records
its own implementation choices, data boundaries and evaluation rules. The
available evidence is intended to be independently reproducible; it does not
claim to reconstruct any private training recipe.

Cross-entropy and Brier are established proper scoring rules. LoRA, shared
candidate heads, consistency regularization and temperature fitting are also
established techniques. Combining them is an engineering baseline. The intended
research contribution is explaining and measuring their effects under controlled
conditions, and deriving improvements supported by those measurements.

The current Typed Decisions pilot instantiates that separation on the real 2B
decoder: hard supervision, full teacher distributions, ordered RPS, and a
Gaussian score-function estimator with an independent leave-one-out baseline
are matched from the same initialization. Thirteen cases, one seed and eight
updates are enough to validate the path, not to choose a method. The next
registered step is four held-out-workflow folds and seeds 42/43/44.

## Questions

1. Does training with CE + Brier improve probability quality beyond CE alone and
   beyond fitting a temperature after training?
2. Does a consistency penalty add stability beyond merely training on shuffled
   candidates twice?
3. For a known differentiable decision reward, what changes when estimating its
   gradient by a score-function policy gradient versus reparameterization?
4. Which improvements transfer to a whole unseen task, rather than only to
   examples of a trained task?
5. How much useful coverage is available at a specified error rate?

## Frozen local ablation protocol v1

| Arm | Training score | Views per microbatch | JS weight |
|---|---|---:|---:|
| A | CE | 2 | 0 |
| B | CE + 0.5 Brier | 2 | 0 |
| C | CE + 0.5 Brier | 2 | 0.05 |

All arms start from the same original EuroBERT checkpoint and fresh adapter/head
initialization. They use the same 384 train, 96 validation and 192 diagnostic
test examples, 2 epochs, 96 updates, candidate permutations, dropout settings,
optimizer and learning rates. Each gets two shuffled views even with no JS term,
so augmentation exposure and encoder-forward count are held fixed.

Subsets are balanced by source/family/label. Their class proportions need not
match deployment prevalence; confidence thresholds are diagnostic, not
validated deployment policies.

Dropout remains enabled independently in both views. C's JS term therefore
penalizes disagreement from both candidate order and dropout. C vs B measures
the added term under this recipe, not a pure causal separation of those two
sources. A future fixed-order/dropout-only control would separate them.

Initial trainable weights, selected IDs and the complete training permutation
plan are hashed and compared before accepting the study. Source and protocol
hashes are saved before its first training run. This is a **local protocol
freeze**, not independent or externally preregistered research.

Report raw and validation-temperature-scaled accuracy, NLL, Brier, ECE, selective
coverage and the same 24-example permutation audit for every arm. Fixed final
epochs are used; test results do not select checkpoints. Source-level results
are primary; mixed-task averages can obscure weak transfer.

This first study uses **one seed and a small sample**. Its test examples were
already inspected in the initial pilot. GoEmotions remains absent from training
and calibration, but now serves as a development diagnostic. Results cannot
establish a general zero-shot advantage or a statistically robust algorithmic
improvement. Future studies need multiple seeds and a new untouched task suite.

## Matched-gradient experiment

Define `J(mu)=E_epsilon[L(softmax(mu+sigma*epsilon),y)]` with standard Gaussian
epsilon and `L=CE+0.5*Brier`. Both estimators target this **same smoothed risk**:

- Score function: `(L(z)-b) * (z-mu)/sigma²`, with independent leave-one-out
  baseline `b` and fixed sigma.
- Pathwise: differentiate `L(mu+sigma*epsilon)` through the sampled action.

The baseline must not include the current sample's reward. We do not normalize
rewards by their sampled standard deviation because that would change the
estimator comparison. The purpose here is to isolate gradient estimation under
a fixed, explicit normalization rule.

Vary candidate count (3/8/16), noise (0.1/0.4), samples (4/16/64), and uncertain
versus confidently wrong logits. Use shared noise between estimators and a large
independent reference sample. Measure variance, mean estimation error and
wrong-direction frequency. Include finite-difference and Monte Carlo checks.

This is a synthetic optimization experiment. Smoothed risk is not identical to
the clean-logit supervised objective. Lower estimator variance does not by
itself prove higher model accuracy, calibrated OOD probabilities, or that RL is
unnecessary when rewards are non-differentiable or delayed.

Gradient-estimator theory is established work; see
[Schulman et al., Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254).
Our contribution at this stage is the explicit controlled experiment, not a
new gradient estimator.

A supplemental control projects the score estimate onto the zero-sum logit
subspace, since softmax is invariant to a common additive logit shift. This
removes irrelevant variance without changing the expected gradient. It is a
basic variance-reduction check, not an exhaustive best policy-gradient baseline.

## Noise and inference probabilities

A separate binary population-CE experiment asks what a noisy-logit objective
actually calibrates. With two independently perturbed logits, its optimum
satisfies `E[sigmoid(d + epsilon_1 - epsilon_2)] = p`, not necessarily
`sigmoid(d) = p`. We solve that monotone condition by bisection and compare
96/192-point Gauss-Hermite quadrature. This toy counterexample is not evidence
about any undisclosed training recipe. It explains why inference must specify
whether it returns unperturbed probabilities or an expectation over noise.

## Does group normalization preserve the optimum?

At the same exactly specified binary noisy-CE optimum, compare a leave-one-out
score estimator, within-group mean centering, additional division by within-group
standard deviation, and pathwise gradients. Integrate the two possible labels
using their known true probability, **after** performing any normalization
separately for each observed label. Use shared samples and 16,384 independent
groups for each of 18 settings.

For a uniform fixed group size, mean centering alone multiplies the expected
score gradient by `(G-1)/G`; it does not move a zero-gradient point. The random
standard deviation can change the update's expected direction, not just its
variance. This experiment checks local stationarity, not full model training.
It omits clipping and auxiliary CE anchors and is not an end-to-end audit of an
external implementation.

## Reproduce

```bash
python -m unittest discover -s tests -v
python scripts/download_base.py --output models/eurobert-2.1b
python scripts/gradient_study.py --output runs/gradient-estimators-v1
python scripts/gradient_study.py --projection-control --output runs/gradient-projected-v1
python scripts/noise_optimum_study.py --output runs/noise-optimum-v1
python scripts/normalization_study.py --output runs/normalization-v1
python scripts/run_ablation.py --output runs/calibration-study-v1 --device mps
python scripts/study_report.py
```

Each experiment requires a new output directory. The report command reads the
completed experiments and replaces only derived report/evidence files.
Model experiments run serially on
one device. The public base is expected at `models/eurobert-2.1b`. Preparation
and installation are documented in the root README.

## Subsequent model-level gradient study

The later `gradient-training-v1` study implements both estimators through the
actual EuroBERT/LoRA training graph, starting from the same independent-scoring
checkpoint. A clean-risk arm separates the noise effect from the estimator
comparison. The protocol fixes a short 192-example / 24-update continuation;
48 examples introduce SNLI and 144 replay old tasks. Sample IDs, initial weights,
encoder work and B/C noise draws are matched. See
[the design](GRADIENT_TRAINING_DESIGN.zh-CN.md) and
[the measured report](GRADIENT_TRAINING_STUDY.zh-CN.md).

All arms trigger norm clipping on every update, so ideal gradient unbiasedness
does not describe the whole AdamW/clipped optimizer. Inference uses clean logits;
a separately labeled [post-hoc probability diagnostic](NOISY_INFERENCE.zh-CN.md)
checks averaged noisy probabilities without changing primary model selection.
