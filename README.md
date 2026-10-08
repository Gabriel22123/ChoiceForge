# ChoiceForge — public research preview

**A task-conditioned decision model for bounded candidate decisions.**

Give it a task, context and candidate descriptions. It returns a candidate ID and
probabilities, without generating reasoning text or arbitrary JSON. Candidate
IDs and the number of candidates can change between requests. Constraint
assessment, intent routing, tool selection and other bounded decisions use the
same interface.

The goal is zero-shot transfer to new tasks without task-specific fine-tuning.
**Universal zero-shot competence is a research goal, not an achieved capability.**
**ChoiceForge** is the project name. The current Python import and CLI remain
`decision_model` and `decision-model` so the research artifacts stay
reproducible while the public API is still alpha.

Before publishing or mirroring this source tree, read the
[publication review](PUBLICATION_REVIEW.md), which fixes the included and
excluded artifact boundary.

<!-- release-status:start -->
**The public source and verified ChoiceForge parameter bundle are released.** Three frozen portfolio seeds preserve every parent decision on the sealed SciFact family while lowering cross entropy by 0.0730–0.0739 and Brier by 0.0149–0.0150. The canonical seed-42 portfolio is exactly fused into one 14 MB residual adapter and packaged with its parent decision parameters as a 55 MB bundle under `models/choiceforge-local-bundle-v1/`. The public 2B base is downloaded separately from its upstream repository. See the [sealed result](docs/SCIFACT_FINAL_EVALUATION_V1_RESULT.zh-CN.md), [adapter result](docs/CANONICAL_MERGED_ADAPTER_V1_RESULT.zh-CN.md), and [publication review](PUBLICATION_REVIEW.md).
<!-- release-status:end -->

## Quick start

ChoiceForge is a structured candidate scorer. It receives a task, optional context,
and a variable-length list of candidates. The runtime returns one candidate ID,
probabilities, a calibration label, and an explicit review flag. The model never
has to generate reasoning text or free-form JSON; the schema is assembled and
validated by the runtime.

```bash
git clone https://github.com/Gabriel22123/ChoiceForge.git
cd ChoiceForge
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model]'
python scripts/fetch_decoder_base.py --output models/minicpm5-2b-base

choiceforge predict-merged \
  --bundle models/choiceforge-local-bundle-v1 \
  --base-path models/minicpm5-2b-base \
  --input examples/request.json \
  --device cpu
```

The command is deterministic for a fixed base revision, bundle, device and input.
The loader verifies the base lock, every bundle file, and the adapter protocol
before inference. All predictions are marked `requires_review: true`; this is a
decision aid, not an autonomous action executor.

## What we built

- A task-conditioned scalar scorer that supports arbitrary candidate counts and
  candidate IDs instead of a fixed classifier label set.
- A decoder route based on the public MiniCPM5-2B base, compared against the
  same-budget EuroBERT-2.1B encoder route under matched public-data protocols.
- An RLCD-inspired outcome-learning pipeline: public outcome targets, proper
  scoring, full/null-context advantage, source-specialist experts, a
  source-balanced portfolio, and a decision-stable projection that preserves the
  parent top choice while improving probability quality.
- A single merged residual adapter so deployment does not need a runtime expert
  router. The merge was checked against all 5,765 cached public rows with a
  maximum probability difference of 1.43e-6.
- A strict, code-owned output contract. Candidate order is audited, JSON is
  schema-checked, and generated chain-of-thought or arbitrary keys are outside
  the serving path.

## What the evidence says

On the preregistered three-seed SciFact evaluation (209 answer-blind-selected
public claim/evidence pairs), every seed kept the parent's hard choice while
improving cross entropy by about 0.073, Brier by about 0.015, and the proper
loss/context-advantage measures. In the matched architecture study, the
MiniCPM decoder reached 81.2–81.8% ARC held-out accuracy versus 33.1–35.7% for
the EuroBERT route under the tested protocol. These are bounded public-data
experiments, not a claim of universal zero-shot competence.

## Train it yourself

The minimal public-data path uses the checked-in release candidate:

```bash
python -m pip install -e '.[data,model]'
decision-model train \
  --data data/release-candidate-v1/expanded.jsonl \
  --config configs/eurobert-public-pilot.json \
  --output runs/my-public-adapter \
  --device auto
```

For the full research route, prepare the public sources, pin their hashes, and
run the staged scripts in this order:

1. prepare and audit public sources (`scripts/prepare_*.py`,
   `scripts/audit_release_data.py`);
2. train source specialists and cache their public features
   (`scripts/run_public_specialist.py` and `scripts/cache_*specialist*.py`);
3. train the decision-stable portfolio
   (`scripts/train_full_expert_portfolio.py`);
4. build and verify the merged adapter
   (`scripts/build_canonical_merged_adapter.py` and
   `scripts/verify_release_bundle.py`).

Every experiment writes its config, seed, selected IDs, hashes and metrics into
its output directory. To continue from a checkpoint, use `--init-from`; this
loads adapter/head parameters but starts a new optimizer and refits calibration.
The [research guide](docs/RESEARCH.zh-CN.md), [algorithm notes](docs/ALGORITHM.md),
and [local bundle guide](docs/LOCAL_MODEL_BUNDLE_V1.zh-CN.md) explain the exact
protocols and expected files.

## Public-data and release boundary

The repository contains only source code, documentation, public-data manifests
and the ChoiceForge parameter bundle trained from public data. It contains no
company data, internal documents, private logs, internal links, credentials or
third-party base checkpoint. The MiniCPM5-2B base is fetched from its public
upstream revision and remains subject to that model's license. See
[THIRD_PARTY.md](THIRD_PARTY.md), [MODEL_CARD.md](MODEL_CARD.md), and
[PUBLICATION_REVIEW.md](PUBLICATION_REVIEW.md) before redistributing weights or
training on new data.

The current endpoint keeps the parent's hard choice by construction and improves
probability quality on one untouched task family. This is evidence of stable
cross-task transfer, not proof of universal competence. Every output remains
marked for human review.

**Historical retention screen:** replay weighting alone and replay weighting plus a
context-advantage hinge were trained from the same initialization and request
plan. The context arm improves the weighting-only endpoint (72.17% vs 71.66%
seen accuracy; 0.847 vs 0.874 NLL), but remains 1.96 points below the parent
Expanded endpoint and regresses SNLI by 5.21 points versus Control. Its formal
fallacy context advantage is still negative. Neither single-seed arm advances;
the consumed diagnostic suite is development-only, and any later release claim
requires newly sealed tasks. See the [screen report](docs/RETENTION_CONTEXT_SCREEN.zh-CN.md).

**Warm-start functional retention improves five of six frozen development
conditions but does not advance.** Both matched arms reach 75.65% seen accuracy,
1.53 points above the parent Expanded endpoint. Adding a KL anchor preserves
that accuracy while improving seen NLL from 0.835 to 0.825 and the consumed
development diagnostic from 65.63% to 66.15%. The remaining failure is causal
judgment context advantage, which stays negative (-0.0136), so no multi-seed or
release claim is opened. See the
[functional-retention report](docs/FUNCTIONAL_RETENTION_SCREEN.zh-CN.md) and
[aggregate failure diagnosis](docs/CAUSAL_CONTEXT_DIAGNOSTIC.zh-CN.md).

**The BoolQ context-repair screen does not advance.** Starting from the
functional-retention endpoint, it adds 512 previously unused public BoolQ rows
with class-balanced weights. The held-out BoolQ slice improves from 72.66% to
74.22%, and its predictions change from an all-yes prior to 64 yes / 64 no.
The +1.56-point gain misses the frozen +2-point threshold, seen NLL worsens by
0.0128, and causal-judgment context advantage remains negative (-0.0197).
This separates prior repair from cross-task evidence use: balancing the former
does not establish the latter. See the
[BoolQ screen](docs/BOOLQ_CONTEXT_REPAIR.zh-CN.md).

**Counterfactual evidence flips are learnable but still do not transfer.** A
deterministic project-authored generator holds the task, claim and Yes/No
candidates fixed while changing only the evidence. The candidate improves from
88.28% to 100% on validation subjects absent from training, and complete-pair
accuracy rises from 76.56% to 100%. Yet consumed natural-task accuracy falls by
1.04 points, causal-judgment context advantage worsens to -0.0628, and seen NLL
worsens by 0.0187. This is within-format rule learning rather than general
evidence use, so the arm does not advance. See the
[counterfactual-evidence screen](docs/COUNTERFACTUAL_EVIDENCE_SCREEN.zh-CN.md).

**Natural eight-way science evidence produces the strongest task-local gain so
far, but still does not preserve the full decision function.** A frozen QASC
screen adds 512 CC-BY-4.0 training questions while hiding source answer letters
and balancing the answer-blind candidate positions. Held-out accuracy rises
from 82.03% to 93.75%, NLL falls from 0.601 to 0.167, and context advantage
rises from 1.540 to 2.004. All five QASC gates pass. Seen NLL nevertheless
worsens by 0.0435, consumed-development accuracy falls by 0.78 points, and
causal-judgment context advantage remains negative. Only 6/11 frozen gates
pass, so this single-seed endpoint does not advance. See the
[QASC evidence screen](docs/QASC_EVIDENCE_SCREEN.zh-CN.md).

**Scaling the same QASC parameter update does not recover retention.** Fixed
25%, 50% and 75% interpolations all keep at least a 3.91-point QASC gain and
pass permutation/ID audits. The 25% point narrowly keeps seen NLL within the
+0.01 bound, but consumed-development accuracy misses by 0.02 points and causal
context advantage drops by 0.114. Larger steps worsen seen NLL and push that
causal drop to 0.132/0.152. No interpolation is selected for training
replication; the next method must preserve both evidence-conditioned and
context-ablated functions. See the
[retention-path diagnostic](docs/QASC_RETENTION_PATH.zh-CN.md).

**Distilling the context-ablated parent function reduces drift but does not fix
cross-task evidence response.** Two matched half-weight QASC runs share the
same parent, data and presentation plan. Adding null-context KL leaves QASC
accuracy at 90.63%, improves seen NLL from 0.8569 to 0.8474, and raises the
consumed-development accuracy from 65.36% to 65.63%. Both absolute bounds still
fail, and causal context advantage changes from -0.03921 to -0.03960 instead of
the registered +0.01 improvement. The method is retained as a documented
partial regularizer, not selected as the next route. See the
[dual-function screen](docs/QASC_DUAL_FUNCTION_SCREEN.zh-CN.md).

**A zero-initialized residual expert isolates parent parameters, but a global
additive correction still changes the composed function.** The parent LoRA and
readout remain bit-identical across all 454 tensors while only a 528,897-
parameter residual head enters the optimizer. Seen accuracy and NLL both pass,
QASC NLL improves by 0.0594 and QASC context advantage rises by 0.1627. The
QASC accuracy gain is only 2.34 points against a frozen 3-point gate;
consumed-development accuracy falls by 0.521 points and causal context
advantage falls by 0.0111. The mechanism passes 8/11 gates and does not advance.
This separates parameter preservation from behavioral preservation: a frozen
parent is insufficient when an unrestricted residual is added to every task.
See the [residual-expert screen](docs/QASC_RESIDUAL_EXPERT_SCREEN.zh-CN.md).

**Joint sparse routing collapses before the residual expert becomes useful.**
The request router is candidate-order invariant and reads only frozen parent
features, while the residual starts at exact zero. All old-task retention gates
pass: seen and consumed-development accuracy are unchanged, seen NLL improves
by 0.0001, and every development context-advantage delta stays within 0.00006.
The one-sided replay sparsity term drives the gate from 0.25 to 0.00388 on
replay and 0.00499 on QASC, however. QASC accuracy is unchanged and its NLL
improves by only 0.0003. The arm fails four of nine gates and does not advance.
The next consumed mechanism screen separates expert learning from router
learning so the router evaluates a nonzero expert. See the
[routed-residual screen](docs/QASC_ROUTED_RESIDUAL_SCREEN.zh-CN.md).

**Staging expert learning before router learning fixes QASC acquisition, while
the unconstrained router saturates open.** The expert-only and router-only
phases preserve their complementary parameter sets exactly. QASC accuracy rises
from 82.03% to 92.97%, NLL falls by 0.3511 and context advantage rises by
0.2202; seen accuracy/NLL also pass. Both replay and QASC gates converge above
0.997, so the router does not separate them. Consumed-development accuracy
falls by 0.78 points and movie-recommendation context advantage by 0.00834.
The arm passes 8/11 gates and does not advance. Training-row counterfactual
grid search nevertheless yields mean optimal gates of 0.266 for replay and
0.835 for QASC, motivating a behavior-derived route teacher. See the
[staged-router screen](docs/QASC_STAGED_ROUTER_SCREEN.zh-CN.md).

**Behavior-derived routing passes 11/12 gates.** A 21-point counterfactual
grid combines gold proper loss, full/null teacher KL and context advantage into
one optimal gate per training request. The source-blind router learns a replay/
QASC gate gap of 0.326, preserves consumed-development accuracy exactly, raises
QASC accuracy by 7.81 points and lowers its NLL by 0.2947. The sole failure is
movie-recommendation context retention at -0.00552 versus a -0.005 bound. Full
and null views receive materially different gates (0.478 versus 0.380 on that
source), identifying paired route consistency as the next mechanism test. See
the [counterfactual-router screen](docs/QASC_COUNTERFACTUAL_ROUTER_SCREEN.zh-CN.md).

**A soft paired-consistency penalty improves but does not clear the final
context bound.** Keeping all other controls fixed, it reduces the consumed
development full/null gate gap to 0.0822 and improves the movie context delta
from -0.00552 to -0.00532. The frozen limit is -0.005, so the screen still
passes only 11/12 gates. Increasing that penalty post hoc would tune to a
consumed decimal margin. The next structural test computes one route from the
context-ablated task/candidate view and reuses it for both branches, making the
route invariant exact. See the
[paired-consistency screen](docs/QASC_CONSISTENT_ROUTER_SCREEN.zh-CN.md).

**A structurally shared context-ablated route clears context retention but
still flips three consumed-development decisions.** One gate is computed from
the null-context task/candidate view and reused for both paths, making the gate
difference exactly zero. All context, seen and QASC gates pass; QASC accuracy
rises 9.38 points. Development accuracy falls from 66.15% to 65.36% because
two causal and one formal-fallacy decisions change, despite better aggregate
NLL. The screen passes 12/13 checks. The route teacher next needs an explicit
parent top-choice preservation cost on replay rows. See the
[shared-route screen](docs/QASC_SHARED_ROUTE_SCREEN.zh-CN.md).

**Adding a hard parent-choice cost on replay does not transfer decision
stability to consumed development tasks.** It lowers the replay target/learned
gate means to 0.250/0.334 and restores exact seen accuracy while keeping the
QASC gain at 9.38 points. Consumed-development accuracy falls by 1.04 points,
however, with three causal and one formal-fallacy errors. Context retention
still passes. The screen passes 12/13 checks and does not advance. Further
constraint stacking on this consumed suite is stopped; a later route needs new
task-family evidence rather than tuning these four decisions. See the
[decision-stable route screen](docs/QASC_DECISION_STABLE_ROUTE_SCREEN.zh-CN.md).

**An exact binary parent-versus-expert route isolates the remaining failure.**
The learned router closes every replay and seen request, opens every QASC
request, preserves the parent exactly when closed, and raises QASC accuracy by
10.94 points. It also opens all 128 requests from one consumed formal-reasoning
family, however, losing three correct decisions and missing the development
gate by 0.28 points. The other two development families remain exactly on the
parent path. The 14/15 result shows that continuous residual leakage is no
longer the blocker; route generalization to unseen task families is. The next
router study must add public task-family diversity and leave-family-out
validation instead of adjusting this consumed threshold. See the
[binary route screen](docs/QASC_BINARY_ROUTE_SCREEN.zh-CN.md).

**Source balancing does not make the semantic router transfer to an unseen
family.** A frozen 7-family leave-one-out study runs both row-weighted and
source-balanced routing for three seeds per fold. Source balancing reaches only
0.5105 macro balanced accuracy and worsens macro Brier from 0.2887 to 0.2987.
Six of seven held-out families are routed almost entirely open or closed. The
router is therefore learning task-description identity more than transferable
per-request expert utility. This diagnostic blocks another semantic-router
endpoint and redirects the next mechanism toward inference-available behavior
signals such as parent confidence, expert shift and decision flips. See the
[leave-family-out result](docs/BINARY_ROUTE_LOFO_RESULT.zh-CN.md).

**Inference-time behavior features improve route calibration but not the hard
decision boundary.** Replacing semantic hidden representations with 24
candidate-order-invariant parent/expert behavior features lowers leave-family-
out macro Brier from 0.2887 to 0.2445 and beats the 0.3195 constant baseline.
The fixed 0.5 decision still reaches only 0.4797 balanced accuracy, however,
and fails per-family noninferiority. In particular, held-out QASC has an 83.4%
open target rate but only a 5.3% predicted open rate. Behavior carries useful
signal, while binary cross-entropy remains tied to family-specific open-rate
priors. The next preregistered route target should regress signed
counterfactual utility with zero as the natural switch point. See the
[behavior-route result](docs/BEHAVIOR_ROUTE_LOFO_RESULT.zh-CN.md).

**Signed counterfactual utility cannot transfer from a single positive-utility
family.** Regressing `asinh(parent cost - expert cost)` gives a natural zero
route boundary, but every one of 42 leave-family-out models predicts the expert
for every held-out row. Macro regret (0.1100) barely differs from always using
the expert and fails to improve the always-parent regret (0.1128) by the frozen
margin. The target audit explains the collapse: QASC is the only family with
positive mean expert utility; all other six have negative mean utility. The
next data revision must add independently sourced positive-utility public task
families before another router objective is justified. See the
[utility-route result](docs/UTILITY_ROUTE_LOFO_RESULT.zh-CN.md).

**The QASC expert is not a general science expert.** On 384 previously consumed
ARC rows it raises accuracy from 80.73% to 81.51%, but mean counterfactual
utility is only +0.0009 and NLL worsens from 0.5442 to 0.5476. ARC-Challenge is
near neutral (+0.0032 utility) and ARC-Easy is negative (-0.0014), so ARC fails
the frozen positive-family screen. Positive-family diversity cannot be created
by applying one task-specific expert to adjacent datasets. The next expansion
must train public family specialists or a family-balanced joint expert, then
measure held-out utility before routing. See the
[ARC expert-utility screen](docs/ARC_EXPERT_UTILITY_SCREEN.zh-CN.md).

**Source-specific experts break the one-positive-family bottleneck.** Using the
same frozen 2B representation and residual-head protocol, three-seed specialists
for SNLI, PAWS-Wiki and CLINC all pass frozen utility, accuracy and NLL gates.
Mean utility is +0.0711, +0.0627 and +0.0153 respectively. Typed Decisions and
JSON Schema fail; notably JSON Schema gains 4.76 accuracy points while worsening
NLL by 0.0750 and producing -0.0968 utility. This shows why routing labels must
use proper counterfactual cost rather than accuracy alone. Together with QASC,
the project now has four consumed positive-utility specialist families. The next
step is a frozen expert-by-family cross-utility matrix before any multi-expert
router is trained. See the
[source-specialist result](docs/CONSUMED_SOURCE_SPECIALISTS_RESULT.zh-CN.md).

**The expert-bank oracle exposes a flaw in scalar route rewards.** Across 724
consumed validation rows, a gold-cost choice among the parent and four canonical
experts raises accuracy from 72.93% to 77.90% and lowers aggregate NLL by 0.0202.
However, the weighted NLL/Brier/context cost selects the high-variance CLINC
expert on 484 rows and worsens Typed Decisions NLL by 0.2447. The bank is useful,
but a learned router must not imitate this scalar oracle directly. The next route
target will apply hard per-row safety constraints first, then optimize utility
inside the eligible set. See the
[cross-utility matrix](docs/SPECIALIST_CROSS_MATRIX_RESULT.zh-CN.md).

The first constrained oracle catches another contract boundary. It records zero
hard-label correctness, NLL, Brier or context regressions and improves aggregate
accuracy by 4.14 points, but Typed Decisions soft cross entropy still worsens by
0.1481. Those rows carry public `target_probabilities`; protecting only their
argmax label is insufficient. This v1 oracle fails its per-source NLL gate. The
next safety rule must compute cross entropy, Brier and context advantage against
the complete target distribution, reducing to one-hot only for hard-label rows.
See the [safe-oracle result](docs/SAFE_SPECIALIST_ORACLE_RESULT.zh-CN.md).

**Full-distribution safety fixes that failure.** Replacing one-hot statistics
with public `target_probabilities` where available yields zero per-row
correctness, cross-entropy, Brier or expected-context regressions and no
source-level accuracy or cross-entropy regression. The constrained oracle raises
accuracy from 72.93% to 77.21%, lowers NLL from 0.8218 to 0.6565 and improves
proper loss by 0.1996 on average. All frozen gates pass. This feasibility-first,
reward-second target is now the required supervision rule for multi-expert route
learning. See the
[distribution-safe result](docs/DISTRIBUTION_SAFE_SPECIALIST_ORACLE_RESULT.zh-CN.md).

The independently trained ARC specialist is directionally useful but does not
pass its frozen gate. All three seeds gain 0.78 accuracy points, mean NLL falls
by 0.0166 and mean utility is +0.0209. One seed reaches only +0.0196 utility,
and the accuracy and NLL gains miss the preregistered +1-point and -0.02 bounds.
Unlike the transferred QASC expert, both ARC subfamilies improve, confirming the
specialist-training direction without qualifying ARC as an additional stable
positive family. See the [ARC specialist result](docs/ARC_SPECIALIST_RESULT.zh-CN.md).

Directly distilling distribution-safe oracle paths into a five-way classifier
does not preserve the underlying constraints. Leave-one-family-out accuracy
rises by 0.76 points, but cross entropy worsens by 0.0287, 26.70% of selected
paths violate at least one row-level safety constraint, path match is 35.79%
and mean proper-loss gain is negative. The next router therefore separates
per-expert feasibility prediction from utility ranking and keeps the parent as
the fallback whenever no expert is conservatively eligible. See the
[multi-expert route result](docs/SAFE_MULTI_EXPERT_ROUTE_LOFO_RESULT.zh-CN.md).

The independently trained TruthfulQA specialist passes every frozen gate on
all three seeds. Accuracy improves by 1.56–1.95 points, NLL falls by
0.2262–0.2461 and mean counterfactual utility is +0.3189. This adds a positive
specialist family whose mechanism differs from science question answering,
while its 256-row validation split is now consumed and cannot support a fresh
release claim. See the
[TruthfulQA specialist result](docs/TRUTHFULQA_SPECIALIST_RESULT.zh-CN.md).

Factoring safety from utility materially improves zero-shot routing but does not
yet pass the frozen safety gate. Nested source-held-out thresholds turn the
direct router's +0.0287 cross-entropy regression into a 0.0051 reduction and
produce +0.0065 proper-loss gain, while row-level constraint violations fall
from 26.70% to 3.82%. Expert coverage is only 4.24% and selected-expert safety
precision is 55.43%, with most remaining errors on soft-distribution decisions.
The next safety model must predict the individual constraint margins and
abstain outside training-family support. See the
[factored route result](docs/FACTORED_SAFE_ROUTE_LOFO_RESULT.zh-CN.md).

Separately calibrated constraint margins reach the opposite boundary: zero
metric regression and zero safety violations, but they abstain on every heldout
row. The frozen 99% residual bounds therefore produce a safe but useless parent
fallback, failing both coverage and positive-gain gates. This result rules out
claiming success through pure abstention and indicates that probability-only
behavior features lack enough information about unseen supervision structure.
See the [constraint-margin result](docs/CONSTRAINT_MARGIN_ROUTE_LOFO_RESULT.zh-CN.md).

Adding candidate-order-invariant frozen 2B semantic summaries improves the
factored router's expert coverage from 4.24% to 7.64% and selected-expert safety
precision from 55.43% to 77.11%. It still fails: cross entropy worsens by
0.0019, mean proper gain is -0.0016 and the constraint violation rate is 4.88%.
Semantic information is useful, but a single seeded router is not a sufficient
risk control for unseen families. See the
[semantic factored result](docs/SEMANTIC_FACTORED_ROUTE_LOFO_RESULT.zh-CN.md).

Requiring unanimous safety and positive gain across all three frozen seeds
reduces the violation rate to 1.66% and restores a small cross-entropy gain.
It still fails the unchanged endpoint gates: selected-expert safety precision
is 84.00%, coverage is 3.45% and mean proper gain is only +0.0009. Seed
consensus is a useful risk control but cannot replace broader family-level
meta-training. See the
[semantic consensus result](docs/SEMANTIC_CONSENSUS_ROUTE_LOFO_RESULT.zh-CN.md).

Adding the qualified TruthfulQA specialist creates a five-expert, seven-family
bank with materially larger safe headroom. On 980 consumed validation rows, the
full-distribution safe oracle raises accuracy from 67.76% to 72.86%, lowers
cross entropy by 0.2059 and has zero row-level or source-level regressions. All
five experts are selected on some rows, including 123 TruthfulQA selections.
This extended bank passes its frozen integrity and safety checks and is the new
meta-training boundary. See the
[extended expert-bank result](docs/EXTENDED_EXPERT_BANK_RESULT.zh-CN.md).

Training the existing shared semantic consensus router on that larger bank does
not transfer the oracle gain. Safety precision falls to 48.00%, coverage to
2.55% and the violation rate rises to 3.37%. Adding a fifth expert changes the
shared behavior vector and every output jointly, so expert-bank expansion
destabilizes previously learned gates. The next router must use independent
per-expert modules with a fixed parent/expert feature contract. See the
[extended consensus result](docs/EXTENDED_SEMANTIC_CONSENSUS_ROUTE_RESULT.zh-CN.md).

Independent 280-feature expert modules restore nontrivial coverage (6.43%) and
isolate bank expansion, but do not solve cross-family calibration. Safety
precision is 47.62%, violations reach 7.96%, cross entropy worsens by 0.0060
and proper gain is -0.0062. This rules out architectural coupling as the sole
cause: thresholds calibrated on six training families remain unreliable on a
seventh unseen family. See the
[modular route result](docs/MODULAR_EXPERT_ROUTE_RESULT.zh-CN.md).

The exact public-derived Control and Expanded JSONL files are included in the
source boundary for reproducible training. A row-level provenance audit verifies
the seven registered public sources, revisions, licenses and external URLs, and
finds no internal source, local path or blind-suite row. See the
[redistribution audit](docs/RELEASE_DATA_AUDIT.zh-CN.md) and the conditional
[release quickstart](docs/RELEASE_QUICKSTART.md). Blind questions, labels,
logits and per-case predictions remain excluded from release bundles.

## Research focus

**Public calibrated-decision ideas → explicit hypotheses → independent
implementations → reproducible experiments.** We separate scoring objectives
from gradient estimators, test the effect of consistency beyond augmentation,
and examine whether noisy training and clean inference report the same
probabilities. Established components are baselines, not claimed inventions.

See the [research protocol](docs/RESEARCH.md) and
[Chinese explanation](docs/RESEARCH.zh-CN.md). The project is an independent
implementation of publicly describable calibrated-decision objectives; it does
not claim to reproduce any private training recipe.

**Public soft-target path:** the trainer now accepts full candidate probability
targets while preserving the original hard-label path. Pinned Typed Decisions
data is prepared as four leave-one-workflow-out folds, so every workflow is
tested once without any of its train cases entering fitting or calibration. A
matched real MiniCPM 2B pilot now runs hard targets, soft targets, typed RPS and
an independent RLCD-style leave-one-out estimator from the same initialization.
It is a 13-case, one-seed integration/directional check and advances no method.
See the [data and evaluation design](docs/TYPED_DECISIONS_STUDY.zh-CN.md) and
[pilot report](docs/TYPED_DECISIONS_RLCD_PILOT.zh-CN.md).
The four-workflow, three-seed screen on fixed 2B representations is complete.
All 48 heads and validation artifacts were locked before the sealed test was
parsed. Soft targets improve mean unseen-workflow CE by 0.108 and win 9/12
endpoints, but regress the security workflow by 0.01208, beyond the registered
0.01 limit. Ordered RPS wins 8/12; RLCD-LOO wins 1/12 and worsens unseen CE and
Brier. No incremental objective advances to full-LoRA validation, so the
simpler direct-supervision baseline remains. See the
[formal result](docs/TYPED_DECISIONS_OBJECTIVE_SCREEN.zh-CN.md) and
[frozen design](docs/TYPED_DECISIONS_OBJECTIVE_SCREEN_DESIGN.zh-CN.md).

**Executable outcome feedback is now tested separately from teacher
distillation.** Public MBPP programs and deterministic mutants provide real
pass/fail outcomes. After the registered IPS and doubly robust screen failed to
change validation decisions, a fixed-log development follow-up consolidated
only candidate outcomes actually observed through exploration and optimized
probability mass over every tied best candidate. It reaches zero train regret
and is non-worse than both replay baselines on all three validation endpoints,
but strictly beats IPS in only one seed, below the registered two-seed rule.
It does not open the official MBPP test or advance to full LoRA. See the
[design](docs/EXECUTABLE_OUTCOME_RLCD_DESIGN.zh-CN.md),
[initial screen](docs/EXECUTABLE_OUTCOME_RLCD_RESULTS.zh-CN.md), and
[feedback-consolidation result](docs/OUTCOME_FEEDBACK_CONSOLIDATION_RESULTS.zh-CN.md).

**The three-seed context-advantage constraint does not advance.** It reduces
the registered margin shortfall in every seed and gives a small aggregate gain
on newly frozen BIG-bench `disambiguation_qa`, but mean gold log-probability gain
falls in every seed and the seen-task accuracy/NLL conditions fail. It is not a
default training objective. See the
[frozen report](docs/CONTEXT_ADVANTAGE_RESULTS.zh-CN.md).

**Latest: same-data premise-group comparison replicated at seeds 42/43.** Both arms
use 768 original public SNLI examples arranged into 256 complete premise groups,
plus the same 256 replay examples, for two epochs/256 updates. Every update has
two examples per relation label and the same replay positions. Grouped versus
dispersed SNLI is **146/192 vs 131/192** at seed42 and **115/192 vs 97/192** at
seed43: +7.8 and +9.4 points, with a descriptive two-seed mean of 68.0% vs 59.4%.
Mean neutral recall is 66.4% vs 63.3%, and precision is 55.7% vs 50.5%.
Seed43 still gets only 16/64 entailment cases right, so relation balance remains
unstable. See the [two-seed report](docs/RELATION_GROUP_MULTISEED.zh-CN.md),
[seed42 details](docs/RELATION_GROUP_RESULTS.zh-CN.md) and
[frozen design](docs/RELATION_GROUP_DESIGN.zh-CN.md). The new data, update-level
label balance and extra epoch also differ from historical studies; historical
gains cannot be attributed only to grouping. Grouping changes gradient
aggregation, with no cross-example attention or new contrastive objective.

Reproduce with `scripts/prepare_relation_groups.py`,
`scripts/run_relation_group_study.py --prepare-only`, then the same runner with
`--device mps`, and `scripts/relation_group_report.py`. The pinned official
archive, earlier public data and D-independent starting checkpoint are required.
Use `--seed 43 --output runs/relation-group-seed43-v1` for the replication, then
run `scripts/relation_group_multiseed_report.py`. All four adapters remain local.
The unchanged authored evidence-flip diagnostic favors dispersed at seed42
(14/18, 2/6 complete vs 15/18, 4/6) and grouped at seed43 (13/18, 1/6 vs
11/18, 0/6). Its direction does not replicate. These already-inspected project
diagnostics are not a fresh benchmark. Two seeds support further study of
grouping, not a stable universal default or zero-shot claim.

**New: the extra epoch and update-level label balance are now isolated.** With
balanced dispersed batches fixed, the second epoch raises SNLI from 76/192 to
131/192 at seed42 and from 74/192 to 97/192 at seed43: +28.6 and +12.0 points.
It adds 128 updates and a repeated exposure, so those two effects remain joined.
With two epochs and every premise/replay position fixed, forcing each update to
contain 2 contradiction, 2 neutral and 2 entailment cases beats naturally varying
label counts by **+32.3 points** at seed42 (131/192 vs 69/192) and **+12.0 points**
at seed43 (97/192 vs 74/192). The varying runs collapse toward different classes
across seeds; balanced batches produce less extreme per-class outcomes. On the
authored evidence-flip probe, varying/balanced scores are 7/18 vs 15/18 at seed42
and 5/18 vs 11/18 at seed43. See the [epoch report](docs/SECOND_EPOCH_RESULTS.zh-CN.md),
[label-balance report](docs/LABEL_BALANCE_RESULTS.zh-CN.md), and
[frozen balance design](docs/LABEL_BALANCE_DESIGN.zh-CN.md). These are two-seed
results on an already inspected trained task, not a universal recipe.

Reproduce the two ablations with `scripts/run_second_epoch_study.py` and
`scripts/run_label_balance_study.py`, followed by their matching report scripts.
The six fixed endpoints can be exported locally with `scripts/export_study.py
--kind relation-ablations`; no remote publication is performed.

**Latest frozen transfer test: the large SNLI gains do not carry over to
WinoGrande.** Before inference, 384 WinoGrande 1.1 development cases were selected
by an input-only hash and all nine endpoints were registered. Balanced versus
variable-label training changes accuracy by +0.3 points at seed42 and -0.3 at
seed43, for a mean of 0.0. Balanced endpoints are only +1.6 and +0.3 points above
the common starting checkpoint, with both paired intervals crossing zero. All
nine models score 49.0%–52.1%, close to binary chance, and balanced training
worsens raw NLL versus variable-label training in both seeds. The models choose
the upstream `option1` content 65.4%–77.9% of the time while its true rate is
50.3%. This is evidence that the current recipe learns the trained relation task
without establishing cross-task commonsense transfer. See the
[frozen blind report](docs/WINOGRANDE_BLIND_RESULTS.zh-CN.md). WinoGrande never
enters fine-tuning or calibration, and its raw text is excluded from bundles.

**Latest public task expansion: PAWS is learnable, but transfer is unstable.**
Two fixed endpoints add 768 PAWS-Wiki training examples plus 256 old-task replay
examples to the seed42/43 balanced-relation checkpoints. PAWS test accuracy moves
from 51.0% to 74.5% at seed42 and 50.0% to 52.6% at seed43. On a newly frozen,
answer-blind 384-case BoolQ subset, accuracy changes by +2.6 and -13.8 points;
on the unchanged WinoGrande set it changes by -0.8 and -0.3 points. The BoolQ
starts predict `yes` on 384/384 and 375/384 cases; PAWS training moves the two
models in different directions. This supports task-specific learning, not stable
cross-task reading comprehension. See the
[task-expansion report](docs/TASK_EXPANSION_RESULTS.zh-CN.md). No PAWS, BoolQ or
WinoGrande text is included in release bundles.

**Common-start task-ratio study: fixed task ratios improve trained-task accuracy
but amplify gradient tails.** Four endpoints use the same starting
weights, 384 PAWS, 384 SNLI and 256 replay rows, one exposure each and 128
updates. Compared with random mixing, fixed `3 PAWS + 3 SNLI + 2 replay` updates
raise SNLI by +2.6/+17.7 points and PAWS by +1.0/+2.1 at seeds 42/43. PAWS raw
NLL worsens slightly in both seeds, while previously inspected BoolQ and
WinoGrande diagnostics change in opposite directions. Maximum pre-clip gradient
norm rises from 111.7 to 8929.9 and from 51.2 to 604.4; every update is clipped.
See the [task-balance report](docs/TASK_BALANCE_RESULTS.zh-CN.md).

**Latest gradient-control study: scale spikes and directional conflict require
separate controls.** Six endpoints compare raw task-gradient aggregation,
median norm capping and deterministic symmetric conflict projection from the
same fixed start, rows and per-seed update plans. Median capping reduces p95
pre-clip norms from 57.53/32.74 to 29.67/29.25, but PAWS changes by -3.6/-0.5
points. Projection improves SNLI by +1.6/+1.0 and overall hard accuracy by
+0.6/+1.4 points, while PAWS changes by +0.5/-3.1 and maximum norms remain
154.59/371.32. All paired PAWS/SNLI bootstrap intervals cross zero. Neither
method is promoted as the default. See the
[gradient-control report](docs/TASK_GRADIENT_CONTROL_RESULTS.zh-CN.md). The next
matched experiment will compare a softer 2x-median cap with cap-then-project;
BoolQ and WinoGrande remain previously inspected diagnostics, not blind tests.

**Follow-up composition study: accuracy stabilizes, but the full advancement
rule still fails.** A 2x-median cap followed by symmetric conflict projection
changes PAWS by +0.5/-0.5 and SNLI by +4.7/+4.2 points at seeds 42/43. The four
primary changes average +2.2 points, but all paired intervals cross zero. The
combined-gradient p95 falls from 57.53 to 31.41 at seed42 and rises from 32.74
to 37.50 at seed43; the WinoGrande candidate-count seed gap also widens from 10
to 15. The method is a promising research candidate, not a default. See the
[composition report](docs/TASK_GRADIENT_COMPOSITION_RESULTS.zh-CN.md).

**Earlier: two-seed matched-update warm-up study completed.** Both recipes use
208 updates and 1,504 example exposures. Repeating 24 inference examples before
the common mixed training reaches **90/192 versus 64/192**, and **103/192 versus
66/192**, compared with a broader 480-example mixed first phase. Raw SNLI NLL
also improves in both seeds. Mean intent, Schema and emotion accuracy decline,
however, and the second small-example seed gets **0/64 neutral cases correct**.
Keep this as a research direction, not a universal default. See the
[full comparison](docs/MATCHED_BUDGET_RESULTS.zh-CN.md). Candidate counts and
lengths differ, so equal updates/exposures do not mean equal FLOPs. One historical
small-example arm is reused; the other three are new runs. Two seeds on an
inspected development set do not establish universal zero-shot competence.
The [authored evidence-flip probe](docs/MATCHED_BUDGET_FLIPS.zh-CN.md) scores
11/18 and 12/18 for small versus 6/18 for both mixed arms; all four still pass
zero of six complete groups. Insufficient-evidence decisions remain unresolved.

Run `scripts/run_matched_budget_study.py --prepare-only`, then the same script
with `--device mps`, then `scripts/matched_budget_report.py`. Earlier public data
and starting checkpoints are prerequisites; see the
[frozen design](docs/MATCHED_BUDGET_DESIGN.zh-CN.md).

**Earlier memorization diagnosis and warm-up continuation.** A fixed
24-example public training set is fit perfectly, but warm-up alone reaches only
66/192 SNLI test decisions and worsens overconfidence. Following the same
1,024-example mixed training then reaches 90/192 (46.9%), versus 64/192 (33.3%)
in the earlier same-seed run. Intent/emotion each lose two correct cases;
Schema gains one. Audited order stability is 25/25. See the
[full results](docs/LEARNABILITY_RESULTS.zh-CN.md). The extra 80 warm-up updates,
balance and dropout changes prevent attributing the gain to curriculum alone.
This single-seed local gain does not establish universal zero-shot competence.
The production readout is unchanged.
The authored [evidence-flip probe](docs/CURRICULUM_FLIPS.zh-CN.md) improves from
6/18 for the earlier same-seed checkpoint to 11/18, but complete groups remain
0/6; insufficient evidence is frequently mistaken for support.

Reproduce with `scripts/overfit_sanity.py`, `scripts/frozen_readout_probe.py`,
`scripts/run_curriculum_probe.py` (each accepts `--device mps`), then
`scripts/learnability_report.py`. The earlier public data and starting checkpoint
are prerequisites; use fresh output directories. Final weights at
`runs/curriculum-probe-v1/continued` use the existing prediction/warm-start API.

**Earlier two-seed public semantic expansion.** Each seed trains on
768 SNLI plus 256 replay examples for 128 updates. On identical test requests,
SNLI changes from 65/192 at initialization to 64/192 and 70/192; sampled intent
changes from 54/61 to 56/61 and 57/61. Strong candidate preferences remain, so
stable relation judgment has not been established. See the
[full report](docs/SEMANTIC_SCALE_STUDY.zh-CN.md) and
[post-hoc diagnosis](docs/SEMANTIC_COLLAPSE.zh-CN.md).
Both new checkpoints pass zero of six complete, project-authored
[evidence-flip groups](docs/EVIDENCE_FLIPS.zh-CN.md). Stable structured outputs
and stable candidate order do not establish evidence-dependent reasoning.

**Completed real-model gradient-estimator study.** Three matched
24-update warm-start runs compare clean proper scoring, noisy pathwise updates
and a score-function estimator of the same noisy risk. SNLI accuracy is
15/45, 11/45 and 15/45 respectively, versus 15/45 at the starting checkpoint.
Lower probability loss did not establish better new-task judgment. See the
[study](docs/GRADIENT_TRAINING_STUDY.zh-CN.md) and
[implementation design](docs/GRADIENT_TRAINING_DESIGN.zh-CN.md).

**Latest architecture study:** independent candidate scoring improves audited
order stability from 4/24 to 24/24 and all-order correctness from 4/24 to 14/24.
Local median request latency rises from 101.6 to 324.1 ms. Schema accuracy
improves, intent/emotion each lose one correct case, and emotion probability
quality worsens. See the [architecture report](docs/ARCHITECTURE_STUDY.zh-CN.md).
Structural stability does not establish general semantic or calibration gains.

**The post-projection trust region did not advance.** All 384 trust updates
across three seeds obey the registered bound: the corrected aggregate norm never
exceeds the same-step raw aggregate norm. However, mean PAWS/SNLI accuracy falls
by 1.0 percentage point versus raw, the worst task/seed change is -7.8 points,
and gradient p95 does not improve consistently. The method controls update
geometry but does not reliably preserve semantic quality. See the
[trust-region results](docs/TASK_GRADIENT_TRUST_RESULTS.zh-CN.md).

**The formal three-seed comparison advances the 2B decoder as the next default
route.** MiniCPM5-2B-Base and EuroBERT-2.1B use the same 1,024 public training
rows, within-seed request plans, 128 updates, final-32-layer rank-8 LoRA and
loss. The decoder passes all eight preregistered conditions across PAWS/SNLI,
retained and task-held-out data, order audits, and ARC-Challenge/Easy opened
only after all six endpoints were fixed. Its ARC total accuracy is
81.2%/81.8%/81.2% across seeds versus 35.7%/35.4%/33.1% for the encoder. Local
median inference is about 10%–11% slower, while sampled device allocation is
slightly lower. See the [results](docs/ENCODER_DECODER_RESULTS.zh-CN.md) and
[comparison design](docs/ENCODER_DECODER_COMPARISON_DESIGN.zh-CN.md). This is
an end-to-end comparison of two specific pretrained bases; tokenizer, corpus
and width differ, so the gap is not a pure causal-versus-bidirectional ablation
or proof of universal zero-shot competence.

**Centered null-context prior subtraction does not advance.** Across the three
fixed decoder endpoints, seeds 42/43 select one shared strength from
`{0,.25,.5,.75,1}` using validation NLL only. Both favor `lambda=0`; equal-seed
mean NLL worsens monotonically from 0.5851 to 0.6773 as subtraction increases.
The preregistered method therefore reduces to no correction and is not a default.
After the selection hash was fixed, 288 fresh BIG-bench `logical_deduction`
cases show useful baseline transfer: across seeds, 3/5/7-choice accuracy ranges
from 78.1%–79.2%, 46.9%–50.0%, and 65.6%–67.7%; all three 12-case order/ID
audits are stable. See the [candidate-prior results](docs/CANDIDATE_PRIOR_RESULTS.zh-CN.md).
This blind set is now consumed and cannot support post-hoc tuning of the method.

**Completed controlled study:** CE → CE+Brier+JS improves sampled intent
ranking from 55/61 to 58/61, but task-held-out emotion accuracy drops from
19/61 to 13/61. Correctness across all audited option orders stays at 4/24.
This single-seed small-sample diagnostic does not establish a general gain.
See [findings](docs/FINDINGS.zh-CN.md) and the [full study](docs/STUDY.zh-CN.md).

**First local pilot:** 384 training examples, 96 optimizer updates. Sampled
intent ranking: 58/61 correct; task-held-out emotion recognition: 13/61;
schema conformance: 16/60. Only 4/24 audited examples kept the same answer under
every tested candidate order. This is a runnable research checkpoint, **not a
validated general-purpose decision model**. See [measured results](docs/RESULTS.zh-CN.md).

## What is implemented

- Original public EuroBERT-2.1B backbone, pinned by revision; fresh LoRA/head.
- Shared scalar candidate scorer: there is no fixed three-class output layer.
- Joint candidate encoding (default) and independent candidate scoring, using
  the same backbone and head parameter set.
- 32-layer LoRA by default; rank/layer count and head size are configurable.
- CE + Brier proper-scoring objective, and semantically aligned consistency
  between independently shuffled candidate orders when enabled.
- Real-model clean, noisy pathwise and Gaussian score-function training options;
  repexternal-projectble noise, separate risk/surrogate logging and gradient-clipping records.
  These are explicit public experiments, not a proprietary RLCD reproduction.
- Pinned, checksummed public data with per-example origin and transformation.
- Training, validation-only temperature fitting, per-source evaluation and
  a whole-task-held-out emotion recognition evaluation.
- Fixed program-built JSON, safetensors checkpoints and warm-start training.

## Start with the CPU experiments (no backbone download)

```bash
python -m pip install -e . torch numpy
python scripts/gradient_study.py --output runs/gradient-estimators-v1
python scripts/noise_optimum_study.py --output runs/noise-optimum-v1
python scripts/normalization_study.py --output runs/normalization-v1
```

These small synthetic experiments examine gradient variance, noisy-training vs
clean-inference probabilities, and whether group normalization preserves an
optimum. They require neither private data nor a trained model. They do not
measure model accuracy. Use a new output directory for each run.

## Setup

Python 3.11+ and Git are sufficient for the source project. Model execution
requires PyTorch and a supported native EuroBERT implementation in Transformers.
CUDA, Apple MPS and CPU are supported by the device selection code; the current
local run uses Apple MPS. CPU/CUDA throughput has not been benchmarked here.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,model]'
python -m unittest discover -s tests -v
```

`requirements-tested.txt` records the direct dependency versions used for the
local run; install with `-c requirements-tested.txt` to use those versions. It
is a tested constraint set, not a cross-platform lock of every transitive wheel.
`requirements-local-lock.txt` also records the full installed dependency set on
the local macOS/Python environment; other platforms may need different wheels.

The native Transformers implementation is used with `trust_remote_code=False`.
The original public weights are about 9.6 GB on disk. MPS/CUDA execution loads the
backbone in FP16 and trains FP32 adapters/head; CPU uses FP32. Initial download
needs internet access. LoRA checkpoints do not contain the full backbone.

To prepare the exact local base used by the experiment scripts:

```bash
python scripts/download_base.py --output models/eurobert-2.1b
```

This downloads only the five files listed in the packaged base lock at its
fixed public revision, then checks their SHA-256 hashes. An already complete
checkpoint is verified without downloading; add `--offline` to require that.

## Reproduce the public data

```bash
decision-model prepare --output data/public-multitask-v1
decision-model validate-data data/public-multitask-v1/cases.jsonl
```

The source lock pins every input file by Git commit and SHA-256. `--offline`
requires the verified `.cache` files already to be present. A new output
directory is required, so a previous experiment is never silently overwritten.

| Source | License | Use |
|---|---|---|
| JSON Schema Test Suite | MIT | Public rule-conformance examples and bounded evidence ambiguity |
| PyCQA Bandit examples | Apache-2.0 | Narrow code-policy examples; not full application security labels |
| CLINC150 | CC-BY-3.0 | User intent ranking, with 2/4/8 supplied candidates |
| GoEmotions | CC-BY-4.0 for data | Seven-emotion, single-label subset, **test only** |

Sources are credited in [THIRD_PARTY.md](THIRD_PARTY.md); transformations and
limitations are in [docs/DATA.md](docs/DATA.md). Public does not mean unlicensed:
derived datasets retain their upstream licenses and attribution requirements.
No private organizational documents, code, logs, examples or previously
fine-tuned private-data adapters are imported by this project.

An optional [SNLI adapter](docs/SNLI_PREPARATION.md) prepares 3,072 training,
384 validation and 768 test records for the next semantic-training experiment.
The gradient study uses 48 of its training examples; the semantic expansion
uses the same 768 examples in each of two seeds, not 1,536 unique SNLI records. Earlier pilot,
calibration and architecture checkpoints do not use SNLI. Its data license is
CC BY-SA 4.0; preparation counts must not be confused with actual training exposure.

To create the next mixture, run `python scripts/compose_public_data.py` after
both input datasets are prepared. This checks their pinned hashes and preserves
all source/split roles. The gradient-estimator study uses this second mixture.

## Train locally

```bash
decision-model train \
  --data data/public-multitask-v1/cases.jsonl \
  --config configs/eurobert-public-pilot.json \
  --output runs/public-multitask-v1
```

The pilot deliberately caps training to **384 examples for 2 epochs**. It tests
the learning pipeline, not full-scale training. The complete prepared dataset
is larger. To train more, use `configs/eurobert-public-full.json` or edit a copy.
These settings are not an established optimal recipe.

`--base-path /path/to/original/EuroBERT` reuses a local original checkpoint; its
weight SHA-256 must match the public base. `--device mps|cuda|cpu` overrides auto.

Train on your own data, optionally warm-starting a previous public adapter:

```bash
decision-model train --data my-data.jsonl \
  --config configs/eurobert-public-pilot.json \
  --init-from runs/public-multitask-v1 --output runs/my-next-run
```

Warm start loads adapter/head weights and starts a **new optimizer**. It is not
an exact interrupted-run resume. The old calibration is refitted on the new
validation split. Optimizer and RNG states are saved for future resume tooling.

For larger runs, `max_train_evaluation_rows` optionally limits the post-training
training-set probe. It does **not** limit training or validation/test evaluation;
`run.json` records both `selected_ids` and `evaluation_ids`. Omit it or use zero
to evaluate the full selected training set. `require_all_rows: true` makes a
frozen dataset fail if any request exceeds the token budget instead of silently
reducing the sample set.

### Expand semantic training with two seeds

```bash
python scripts/run_semantic_scale_study.py --device mps
python scripts/semantic_scale_report.py
python scripts/semantic_collapse_diagnostic.py
python scripts/evidence_flip_probe.py --device mps
```

The [frozen design](docs/SEMANTIC_SCALE_DESIGN.zh-CN.md) uses 768 SNLI examples
plus 256 old-task replay examples per seed, and compares both final checkpoints
against the same fixed starting weights on identical evaluation requests.
It requires the prepared second public mixture and pinned independent checkpoint.
Both data exposure and update budget increase relative to the starting model;
this is an adaptation study, not an isolated compute-matched data-scaling effect.
SNLI participates in training and therefore is not a zero-shot task evaluation.
The last two commands are post-hoc diagnostics, not additional training or
checkpoint selection. The authored evidence-flip cases are a small diagnostic,
not an external benchmark. Both `runs/semantic-scale-v1/seed-42` and `seed-43`
can be inspected with `predict` or continued with `train --init-from`.

### Compare gradient estimators on the model

```bash
python scripts/run_gradient_training_study.py --device mps
python scripts/gradient_training_report.py
```

This warm-start study requires the previous independent checkpoint and the
second public mixture. It compares clean proper scoring, noisy pathwise gradients
and a Gaussian score-function estimator. B/C share the same sampled risk and
noise plan; C uses an independent leave-one-out baseline, without group reward
standard-deviation normalization. See [the design](docs/GRADIENT_TRAINING_DESIGN.zh-CN.md).

Standalone small-run configurations are
`configs/eurobert-independent-pathwise-pilot.json` and
`configs/eurobert-independent-score-function-pilot.json`. They also support
`train --init-from` with a compatible independent checkpoint.

### Compare candidate encodings

```bash
python scripts/run_architecture_study.py --device mps
python scripts/architecture_report.py
```

This starts two fresh CE-only runs with identical initialization, samples,
option-order plans and optimizer updates. It measures accuracy, order stability,
training work and loaded-model latency. Equal updates do **not** mean equal
compute: independent scoring encodes the context once per candidate.

For a standalone independent run, use
`configs/eurobert-independent-pilot.json`. Each candidate description must be
self-contained: relational choices such as "none of the above" are not
semantically equivalent to the joint mode. Warm start rejects cross-mode
loading. See [encoding design and tradeoffs](docs/CANDIDATE_ENCODING.zh-CN.md).

## Use a trained checkpoint

Use the verified merged model bundle:

```bash
choiceforge predict-merged \
  --bundle models/choiceforge-local-bundle-v1 \
  --base-path models/minicpm5-2b-base \
  --input examples/request.json \
  --device cpu
```

The bundle loader verifies every parent and adapter file before loading, then
verifies the separately downloaded public base against its pinned revision.
See the [local bundle guide](docs/LOCAL_MODEL_BUNDLE_V1.zh-CN.md).

Earlier experimental checkpoints remain available through the original command:

```bash
decision-model predict --checkpoint runs/public-multitask-v1 \
  --input examples/request.json
```

Input:

```json
{
  "task": "Select the tool that can satisfy the user's request.",
  "context": "Remind me tomorrow morning to submit my application.",
  "choices": [
    {"id": "calendar", "description": "Create reminders and scheduled events."},
    {"id": "search", "description": "Search public web pages for information."},
    {"id": "calculator", "description": "Perform arithmetic calculations."}
  ]
}
```

Output always contains `choice_id`, `probabilities`, `requires_review`,
`score_kind` and `calibration`. Probabilities are model estimates, not a guarantee
of correctness. The preview returns `requires_review: true`. Without an explicit
"none of these" choice, the model is forced to rank only the supplied choices.
Business uncertainty options and model confidence are separate concepts.
In independent mode, express any fallback with a self-contained definition;
the scorer does not read the other candidates when interpreting an option.
See the [fixed output contract](docs/OUTPUT_CONTRACT.md) and its packaged JSON
Schema for the exact machine-checked constraints.

The same command also accepts strict typed inputs:

```bash
decision-model predict --checkpoint runs/architecture-comparison-v1/seed42-decoder \
  --base-path models/minicpm5-2b-base --input examples/boolean-request.json

decision-model predict --checkpoint runs/architecture-comparison-v1/seed42-decoder \
  --base-path models/minicpm5-2b-base --input examples/score-request.json
```

Boolean responses add `value` and `probability_true`. Ordered Score responses
add `level_index` and a probability-weighted, zero-based `expected_score`.
All typed variants retain the full distribution and `requires_review: true`;
their bounded union is packaged as `typed-prediction.schema.json`.

To try the independent checkpoint from the completed architecture study:

```bash
decision-model predict --checkpoint runs/architecture-study-v1/D-independent \
  --base-path models/eurobert-2.1b --input examples/request.json
```

This checkpoint selected `calendar` with probability about 0.668 on the reminder
example above; the original pilot selected the wrong tool. This previously
inspected illustration is not a new held-out benchmark or an automatic-action
threshold. The output still requires review.

Inputs support 2–32 choices and are rejected above the configured token budget;
they are not silently truncated. This preview handles text decisions, not image
inputs, free-form generation, live data retrieval, or action execution.

## How to judge progress

`evaluation.json` reports accuracy, random-choice baseline, NLL, Brier score,
calibration error and coverage at confidence thresholds, broken down by source.
**Do not combine seen-task accuracy with held-out-task accuracy and call the
result zero-shot performance.** GoEmotions is absent from both training and
temperature fitting. Its public test subset is task-held-out relative to this
fine-tuning run, not guaranteed absent from backbone pretraining.

Changing option order is a separate evaluation. See `decision-model audit --help`.
Run selection is the fixed final epoch; test scores do not select checkpoints.
Repeatedly tuning on this held-out task turns it into a development task and
requires a new final benchmark.

Next evaluation targets include unseen task families, unseen label descriptions,
paraphrased instructions, no-valid-candidate cases, longer contexts and languages
other than English. Supporting the interface does not establish competence on
those tasks.

## License and references

Original project code: Apache-2.0. Base model: Apache-2.0. Each dataset retains
its own license. See [LICENSE](LICENSE), [THIRD_PARTY.md](THIRD_PARTY.md) and
[docs/ALGORITHM.md](docs/ALGORITHM.md). This is an independent experimental
implementation inspired by public candidate-scoring and calibrated-decision
ideas; it is an independent project with its own implementation and evaluation protocol.
