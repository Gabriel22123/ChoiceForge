# Dynamic candidate decisions and calibrated training

## Architecture

Input = task + context + a sequence of candidate descriptions. Insert the public
tokenizer's mask token before every candidate. A bidirectional EuroBERT encoder
provides a contextual representation at each marker. The **same** small
LayerNorm/MLP scalar head scores every candidate; softmax over the actual choices
produces a variable-length distribution. IDs are used only by program code to
associate scores with the caller's choices.

This follows a public candidate-scoring design pattern with an independently
implemented head. A shared head allows unseen label meanings and different
choice counts; it does not guarantee the encoder can understand every new task.

The alternative `candidate_encoding=independent` encodes task/context with each
candidate separately, using the same head. Raw candidate scores then do not
depend on list position or other candidate descriptions. Inference uses one
candidate per encoder call; exact score ties still follow the first-index rule.
This repeats context work and requires self-contained candidate meanings.

## Training objective

For a hard outcome y and predicted distribution q:

`L_proper = -log(q_y) + beta * sum_k (q_k - 1[k=y])^2`

The default beta is 0.5. In expectation under true distribution p, excess loss
over predicting p equals `KL(p || q) + beta * ||p-q||²`. This makes truthful
probability estimation the population optimum. It does not prove finite-sample
calibration, out-of-distribution reliability or good representation learning.

When a public dataset supplies a probability target p directly, the same code
uses `-sum p_k log(q_k) + beta * sum(q_k-p_k)^2`. Target mass is keyed by
candidate ID and realigned after candidate shuffling. Temperature fitting and
soft CE/Brier/TV evaluation retain the full distribution. Boolean and Choice use
this categorical proper score; ordered Score questions can add normalized ranked
probability score over cumulative masses.

The noisy estimators now support the same hard or distribution targets. For
score-function training, CE/Brier and optional ordered RPS use independently
computed leave-one-out baselines; an optional clean soft-CE anchor is recorded
separately. The matched real-model pilot freezes four arms and reports both the
sampled risk and optimization surrogate. Its one seed and tiny sample validate
implementation only; see `TYPED_DECISIONS_RLCD_PILOT.zh-CN.md`.

The initial pilot independently shuffles candidate order twice. After aligning the
second set of logits by candidate ID, average both proper losses and add
`lambda * JS(q_view1, q_view2)`. Default lambda is 0.05. Padding candidates are
excluded by masked logits. Gradients flow through both views.

The clean baseline uses **exact supervised gradients** for a calibrated-decision
objective. It is an independent implementation and should not be marketed as a
reconstruction of a private training algorithm. Proper scoring reference:
[Gneiting & Raftery, 2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

## Real-model gradient-estimator options

`gradient_estimator` can be `clean` (legacy default), `pathwise`, or
`score_function`. The latter two optimize the same defined Gaussian-smoothed
proper-scoring risk. They draw `noise_samples` actions per request in logit space,
not by rerunning the encoder. A dedicated seeded CPU generator isolates these
draws from dropout; its draw plan is hashed and recorded.

`pathwise` differentiates through the sampled logits and proper loss.
`score_function` detaches the sampled actions/costs and differentiates a Gaussian
log-density surrogate with an independent leave-one-out cost baseline. Since
the objective minimizes cost, the surrogate uses positive cost advantages.
There is no group standard-deviation normalization or PPO ratio clipping.
The real sampled risk is logged separately from the surrogate value.

Both noisy methods can center real candidate logits to remove the common-offset
gradient direction. Padded classes are excluded from the cost and policy density.
Unbiasedness applies before AdamW and gradient-norm clipping; the actual optimizer
does not inherit a guarantee of unbiased updates or calibrated probabilities.
Inference uses clean logits and optional validation-only temperature scaling.
See [the model-training design](GRADIENT_TRAINING_DESIGN.zh-CN.md) for equations,
controls and the distinction between this experiment and a private RLCD recipe.

## Calibration and evaluation

A scalar temperature is fitted with a fixed log-spaced grid using validation
examples only; identity temperature is always a candidate. It preserves argmax
and can improve held-out NLL, but validation calibration does not automatically
transfer to an unseen task. Report raw and calibrated results separately.

Evaluate accuracy relative to 1/number-of-choices, NLL, Brier, ECE and selective
coverage per task. Brier/NLL depend on choice count, so mixed-source averages
need per-source context. Also audit candidate permutations and ID renaming.
The local preview leaves decisions reviewable; no threshold has been validated
for automatic production action.

"Insufficient evidence" or "none of these" may be explicitly defined by a caller
as a candidate. Low model confidence is a separate property of the probability
distribution and is not automatically mapped to an evidence label.
