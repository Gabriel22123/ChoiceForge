# ChoiceForge public-data candidate-decision research checkpoints

## Model identity

- Project name: ChoiceForge.
- Status: local public research preview; the sealed-family gates and independent
  bundle verification pass, but no remote model or repository has been published.
- Selected research route: `openbmb/MiniCPM5-2B-Base`, revision
  `96a57cd572a02506b4500f54427dca24970c1bac`.
- Architecture: causal decoder, independent candidate encoding, final-token
  readout, 32-layer rank-8 LoRA and a shared scalar MLP. No fixed business classes.
- Trainable parameters: 10,097,153. Base weights remain frozen.
- Checkpoint consists of adapters and scoring head, not the full backbone.
<!-- release-status:start -->
- Current endpoint: a 39 MB parent decision checkpoint plus one 14 MB exactly
  merged residual adapter. The local portable bundle is about 55 MB and excludes
  the public 2B base weights.
- On 209 sealed SciFact claim/evidence pairs, all three frozen seeds preserve
  every parent hard choice. Cross entropy improves by 0.0730–0.0739, Brier by
  0.0149–0.0150, proper loss by 0.0805–0.0814, and context advantage by
  0.0809–0.0818.
- The canonical adapter uses the lowest registered seed (42), independent of
  outcome. It matches the six-expert portfolio across 5,765 cached public rows
  within 1.43e-6 and retains every projection step and choice.
<!-- release-status:end -->

- Model artifact SHA-256:
  `8e4c816e43ea70ccac43145646583e6466357455bc794761faecbe0fe23f196a`.
- The sealed result supports probability-quality transfer on one new task
  family. Accuracy is intentionally unchanged by the decision-stable projection;
  this result does not establish universal task competence.

- Latest retention/context screen: neither single-seed arm advances. The
  context-constrained arm reaches 72.17% seen accuracy and 0.847 NLL, but is
  1.96 points below the parent Expanded endpoint, regresses SNLI by 5.21 points
  versus Control, and retains negative context advantage on formal fallacies.
  The consumed diagnostic suite is development-only. See
  `docs/RETENTION_CONTEXT_SCREEN.zh-CN.md`.
- Warm-start functional retention passes five of six frozen conditions but does
  not advance. The KL arm keeps 75.65% seen accuracy, improves seen NLL to
  0.825 and the consumed development diagnostic to 66.15%, but causal-judgment
  context advantage remains negative at -0.0136. See
  `docs/FUNCTIONAL_RETENTION_SCREEN.zh-CN.md` and
  `docs/CAUSAL_CONTEXT_DIAGNOSTIC.zh-CN.md`.
- A subsequent BoolQ context-repair screen also does not advance. Class-balanced
  continuation changes the 128-case BoolQ validation predictions from all-yes
  to 64 yes / 64 no and improves accuracy from 72.66% to 74.22%, but misses the
  frozen +2-point gain, worsens seen NLL by 0.0128, and leaves causal-judgment
  context advantage negative at -0.0197. The BoolQ and causal suites are now
  consumed development evidence, not fresh release evidence. See
  `docs/BOOLQ_CONTEXT_REPAIR.zh-CN.md`.
- A project-authored counterfactual-evidence screen reaches 100% on 128 generated
  validation rows and 64/64 complete flip pairs, versus 88.28% and 49/64 for the
  parent. It does not transfer: consumed natural-task accuracy falls by 1.04
  points, causal context advantage worsens to -0.0628, and seen NLL worsens by
  0.0187. It does not advance. See
  `docs/COUNTERFACTUAL_EVIDENCE_SCREEN.zh-CN.md`.
- A frozen public QASC screen then improves held-out eight-way science accuracy
  from 82.03% to 93.75%, NLL from 0.601 to 0.167, and context advantage from
  1.540 to 2.004. Its five task-local gates all pass, but seen NLL worsens by
  0.0435, consumed-development accuracy falls by 0.78 points, and causal
  context advantage remains negative. It passes 6/11 gates and does not advance
  beyond a single-seed development screen. See
  `docs/QASC_EVIDENCE_SCREEN.zh-CN.md`.
- Fixed 25%, 50% and 75% interpolations between the retained parent and QASC
  endpoint all preserve meaningful QASC gains and structural invariance, but
  none passes the retention path screen. At 25%, seen NLL is only 0.009993
  above the parent, while development accuracy still misses its bound and
  causal context advantage falls by 0.114. The larger points worsen both
  calibration and causal response retention. No interpolated checkpoint is a
  model candidate. See `docs/QASC_RETENTION_PATH.zh-CN.md`.
- A matched half-weight QASC study adds KL preservation of the parent's
  context-ablated function. It keeps QASC accuracy at 90.63%, improves seen NLL
  by 0.00947 versus the evidence-only control and improves consumed-development
  accuracy by 0.26 points. Seen NLL and development accuracy still miss their
  absolute bounds, while causal context advantage changes by -0.00039 versus
  control. The mechanism gate fails and the route is not selected. See
  `docs/QASC_DUAL_FUNCTION_SCREEN.zh-CN.md`.

## Intended research use

Task-conditioned, bounded decisions from caller-supplied candidate descriptions.
Examples include intent routing, semantic selection, and evidence/rule checks.
New candidate IDs and meanings do not require a different output layer.
The interface accepts 2–32 candidates; pilot training contains 2/3/4/8 choices,
and the held-out emotion task has 7. Interface support for 32 choices is not
evidence of tested accuracy at that size.

The public runtime provides strict Boolean, Choice and ordered Score views over
the same candidate model. Boolean criteria become two explicit candidates;
Score levels become 2–10 ordered candidates. Program-built typed responses add
only deterministic summaries such as top-two margin, probability of true and
expected zero-based score. Both the base and typed JSON contracts forbid
free-form reasoning fields and always mark this research preview for review.

Inference returns program-built structured output. It does not generate text,
retrieve facts, execute actions or supply an automatic production gate.

## Training

The initial studies use fresh adapter/head initialization over the original public pretrained base.
384 publicly derived training examples, two epochs, 96 optimizer updates,
batch size 2 with four-step accumulation and two option-order views.
The initial `public-multitask-v1` pilot uses CE + 0.5 Brier + 0.05 aligned
Jensen-Shannon consistency. The controlled study has three independent arms:
`A-ce` (CE), `B-proper` (CE + 0.5 Brier), and `C-consistency` (adding 0.05 JS).
All three use the same two-view exposure and initial trainable parameters.
The subsequent `architecture-study-v1` compares fresh CE-only `J-joint` and
`D-independent` checkpoints with the same parameter initialization and request
plan. The latter encodes each candidate separately with task/context; equal
updates require more encoder work. `model.json` records `candidate_encoding`.
Read each checkpoint's `model.json` and `run.json` for its precise identity;
results from different checkpoints are not interchangeable. Validation temperature
uses 96 examples. The final epoch was selected in advance. Dataset licenses,
versions, transformations and caveats are recorded in `THIRD_PARTY.md`,
`sources.lock.json` and `docs/DATA.md`.

No private organizational fine-tuning dataset or adapter is used. This statement
does not describe or certify the base model's entire pretraining corpus.

### Public probability-target path

The trainer also accepts complete candidate distributions keyed by candidate ID.
`LocalLLaMA/typed-decisions` is pinned at revision
`ea9306458d6e9563628369a3d1e72e362fb381d2` under Apache-2.0 and prepared as
four leave-one-workflow-out folds. In each fold, the held-out workflow's entire
upstream train split is discarded; its official test cases are evaluated only
after fitting. The other three workflows provide train/validation and official
seen-workflow test cases. Every fold contains 3,600 train, 900 validation and
2,000 test questions.

A matched 13-case MiniCPM pilot verifies four real training paths: hard-target
CE+Brier, full-distribution CE+Brier, typed ordered RPS, and a four-action
Gaussian score-function estimator with an independent leave-one-out baseline
and clean CE anchor. All arms share initial trainable parameters, selected rows,
candidate permutations, encoder work and eight updates. This one-seed pilot is
integration/directional evidence only and advances no method. Teacher agreement
is not measured action correctness. No full four-fold/multi-seed checkpoint is
yet selected or presented as an improved model. See
`docs/TYPED_DECISIONS_RLCD_PILOT.zh-CN.md` and
`docs/evidence/typed-decisions-rlcd-pilot-v1.json`.

The subsequent representation-frozen formal screen uses all four held-out
workflows, seeds 42/43/44 and 48 matched heads. Every head and validation
artifact was checksummed before any sealed test file was parsed. Soft targets
beat hard targets on 9/12 held-out endpoints and improve mean held-out soft CE
by 0.1081, but fail the per-workflow rule because security incidents regress by
0.01208 versus a 0.01 limit. Ordered RPS wins 8/12. RLCD-LOO wins 1/12 and
worsens mean held-out CE by 0.01035 and Brier by 0.00539. No incremental method
advances to full-LoRA validation. This is objective-screen evidence on fixed
representations, not a final adapter comparison. See
`docs/TYPED_DECISIONS_OBJECTIVE_SCREEN.zh-CN.md`.

## Evaluation and limits

See `docs/RESULTS.zh-CN.md` for the **initial pilot's** measured results. The held-out emotion task is
absent from fine-tuning and calibration, but may have appeared in pretrained
model corpora. Accuracy is 13/61 on the balanced diagnostic subset. This sample
does not establish reliable transfer above chance across unseen tasks.

Schema accuracy is below the uniform-choice baseline on this diagnostic.
Only 4/24 order-audited examples preserved their answer across all tested
permutations. Option-order sensitivity is a known weakness of the initial joint pilot.
The relatively high intent score uses sampled candidates and cannot be compared
to the original full CLINC150 benchmark or called universal zero-shot accuracy.
These figures refer to the initial pilot only. Each study arm has its own
evaluation and audit records; no winning checkpoint is selected from this
already-inspected, single-seed development diagnostic.

The completed controlled study is reported in `docs/STUDY.zh-CN.md`, with
interpretation in `docs/FINDINGS.zh-CN.md`. Sampled intent accuracy improves
across A/B/C (55/56/58 correct out of 61), while held-out emotion accuracy
declines (19/17/13 out of 61). All three have only 4/24 cases correct under every
audited candidate order. The additional objectives do not establish a general
quality improvement in this study.

The architecture study (`docs/ARCHITECTURE_STUDY.zh-CN.md`) improves audited
stability from 4/24 to 24/24 and all-order correctness from 4/24 to 14/24.
Independent vs joint accuracy is 7/10 vs 5/10 (Bandit), 54/61 vs 55/61 (sampled
intent), 18/61 vs 19/61 (emotion), and 21/60 vs 16/60 (Schema). Emotion NLL
worsens. Local warmed median latency is 324.1 vs 101.6 ms. This remains a
single-seed development diagnostic, not a new blind test or a universal gain.

Independent scoring requires self-contained candidate descriptions. It cannot
interpret an option using the other choices' contents. Exact score ties retain
the first-index tie break; the measured 24/24 stability is not a guarantee for
all possible inputs. Candidate probabilities also depend on the supplied set.

SNLI does not train the earlier pilot, calibration or architecture checkpoints.

### Subsequent gradient-training-v1 checkpoints

`A-clean`, `B-pathwise` and `C-score-function` warm-start the same previous
independent checkpoint with a new optimizer. Each trains for one epoch / 24
updates on 192 records: 48 SNLI and 144 replayed old-task examples. B/C share
Gaussian noise plans (16 draws, sigma 0.4) and optimize the same smoothed
CE+0.5 Brier risk using different gradient estimators. They use centered logits;
C has an independent leave-one-out baseline and no reward-standard-deviation
normalization. Inference remains clean, with validation-only temperature scaling.

On the same new 192-case diagnostic selection, A/B/C obtain 15/11/15 correct
out of 45 SNLI cases, compared with 15 at the starting checkpoint. Emotion
accuracy rises from 14/46 to 19/46 in all arms; probability calibration does not
uniformly improve. C's fitted temperature worsens emotion NLL from 2.2629 to
2.6253. All arms retain 25/25 audited order stability. All 24 updates per arm
trigger gradient-norm clipping; ideal estimator unbiasedness is not a guarantee
for the resulting AdamW/clipped updates. These are single-seed short adaptation
results, without method-specific hyperparameter tuning or a fresh blind test.
See `docs/GRADIENT_TRAINING_STUDY.zh-CN.md` and the separate, post-hoc
`docs/NOISY_INFERENCE.zh-CN.md`. No winning checkpoint is selected for deployment.

### Subsequent semantic-scale-v1 checkpoints

`seed-42` and `seed-43` each warm-start `D-independent`, using the same 768
public SNLI examples plus 256 previously trained old-task replay examples.
Each uses a new optimizer, clean CE + 0.5 Brier, one view, one epoch and 128
updates. The seeds change order and dropout, not starting weights or data IDs.
The 128-row training evaluation probe is smaller than the 1,024-row training set.
Validation has 192 records and test has 384; the starting weights are reevaluated
on exactly these cases, with temperature fitted on the same validation split.

SNLI accuracy is 64/192 and 70/192, versus 65/192 initially. Seed 42 always
selects neutral on this SNLI test; seed 43 also retains a strong label preference.
Sampled intent rises from 54/61 to 56/61 and 57/61; emotion changes from 18/61
to 25/61 and 20/61. Schema changes from 21/60 to 22/60 and 18/60. This does
not establish consistent semantic or universal zero-shot improvement. Both
final checkpoints are retained, with no test-selected winner. SNLI is trained;
GoEmotions is still task-held-out during fine-tuning but repeatedly inspected.
See `docs/SEMANTIC_SCALE_STUDY.zh-CN.md` and the explicitly post-hoc
`docs/SEMANTIC_COLLAPSE.zh-CN.md`. The separate evidence-flip probe uses
18 project-authored cases, never training data, and is not an external benchmark.
All three checkpoints (initial/seed-42/seed-43) pass zero of six complete
evidence-flip groups; individual-case scores are 7/18, 6/18 and 5/18. See
`docs/EVIDENCE_FLIPS.zh-CN.md`. This diagnostic does not support reliable
evidence-dependent decisions, even though output/order contracts remain stable.

### Subsequent curriculum-probe-v1 checkpoint

`continued` starts from a fixed 80-update warm-up on 24 public SNLI training
examples, then runs exactly the previous seed-42 mixed-data recipe: 1,024 rows,
128 updates, restored dropout 0.05 and a fresh optimizer. All warm-up examples
are inside the existing 768 SNLI training rows. No validation/test examples
enter training. The warm-up was gated only by training fit at a fixed endpoint;
both warm-up and continuation are evaluated, without test-based weight selection.

Warm-up fits 24/24 training examples but scores 66/192 on SNLI test with raw
NLL 3.8032. Continuation reaches 90/192 with raw NLL 1.0870, versus 64/192 and
1.0988 for the previous seed-42 control. It predicts entailment 113 times,
neutral 45 times and contradiction 34 times; label bias remains. Intent is
54/61, Schema 23/60, Bandit 5/10 and held-out emotion 23/61. Order stability
is 25/25, with 15/25 correct under all audited orders. The training probe includes
24 repeatedly exposed warm-up examples (16/24 correct) and only eight other
SNLI examples (4/8); it is not a representative estimate of all training rows.

This is a single-seed development result with 80 additional updates. Repetition,
label balance, dropout and training order change together; it is not a matched-
compute attribution to curriculum or an RL-estimator comparison. SNLI is a
trained task, not zero-shot. See `docs/LEARNABILITY_RESULTS.zh-CN.md`. Separate
frozen-feature head probes reach 24/24 with marker or mean features and 8/24
without context; this proves only small-set fitting capacity. Those mean-readout
heads are diagnostics, not deployed checkpoints or an architecture change.
The previously inspected, authored evidence-flip probe scores 7/18 after warm-up
and 11/18 after continuation; neither passes any of the six complete groups.
The first five continued groups correctly distinguish support from contradiction
but misclassify insufficient evidence as support. See `docs/CURRICULUM_FLIPS.zh-CN.md`.

### Subsequent matched-budget-v1 study

Four fixed final checkpoints compare two phase-one data recipes at seeds 42/43:
`small` repeats 24 SNLI training examples 20 times with balanced batches;
`mixed` takes 480 distinct rows from a seeded shuffle of the existing mixed
training set. Both use 80 updates of six examples, dropout zero, then reset the
optimizer and perform identical 128-update, 1,024-example mixed training with
dropout 0.05. Each arm totals 208 updates and 1,504 example exposures. Candidate
counts and token lengths differ; this is not an equal-FLOP comparison. Phase-two
sample/candidate order and encoder work are verified equal within each seed.

`small-42` reuses the completed curriculum checkpoint after weight/source checks;
the other three arms are new. SNLI accuracy is 90/192 vs 64/192 for seed42 and
103/192 vs 66/192 for seed43, with lower raw NLL in both small arms. Mean accuracy
is 50.3% vs 33.9%. Both mixed arms strongly favor neutral; small-43 instead
classifies zero of 64 neutral examples correctly. Mean intent, Schema and emotion
accuracy regress by 2.5, 5.0 and 1.6 percentage points. Code-strategy mean is equal,
with opposite seed-level effects on only ten examples. All four pass 25/25 order
stability checks; all-order correctness is respectively 15/25, 13/25, 13/25, 14/25
for small42, mixed42, small43, mixed43. No winning checkpoint is selected.

The comparison supports further study of focused warm-up, not a general default
or a claim that repetition alone causes improvement: label balance, source
proportions and example diversity also differ. Historical reuse, only two seeds,
and repeatedly inspected evaluation restrict the conclusion. The objective is
fixed clean CE + 0.5 Brier; this study does not compare RL gradient estimators.
See `docs/MATCHED_BUDGET_RESULTS.zh-CN.md` and its machine-readable evidence.
On the unchanged authored evidence-flip diagnostic, small42/small43 score
11/18 and 12/18; mixed42/mixed43 each score 6/18. All pass zero of six complete
groups. Small43 gets all 12 support/contradiction cases right but all six
insufficient-evidence cases wrong. These cases never enter training and are
not a fresh external benchmark. See `docs/MATCHED_BUDGET_FLIPS.zh-CN.md`.

### Subsequent relation-group-v1 study

Two new fixed-final seed42 checkpoints use the same 1,024 training rows, initial
D-independent tensors, two epochs, 256 updates and 2,048 example exposures. The
768 SNLI examples are original official-training rows: 256 normalized premises,
each with entailment, contradiction and neutral labels. Official dev/test
premises, duplicate pairs and conflicting labels are excluded before selection.
The 256 old-task replay rows and all 192 validation/384 development-test rows
are unchanged. Source text, labels, archive indexes and hashes are verified.
704 of the 768 source rows have only one upstream annotation; no consensus
claim or relabeling is made. Normalized deduplication cannot exclude all semantic
near-duplicates. SNLI-derived data retains CC BY-SA 4.0, separate from code.

Each update contains two cases per relation label plus the same two replay
cases at the same positions. Grouped updates contain two complete premise
triples; dispersed updates contain six different premises. This changes gradient
aggregation only, not cross-example attention, the scalar head or the clean
CE + 0.5 Brier objective. Both process 6,596 candidate sequences and 440,576
unpadded tokens; padding work and dropout-mask assignments differ.

Grouped versus dispersed SNLI accuracy is 146/192 (76.0%) versus 131/192
(68.2%). Correct contradiction/neutral/entailment counts are 51/43/52 versus
38/47/46, each out of 64. Neutral recall decreases from 73.4% to 67.2%, while
neutral precision increases from 57.3% to 67.2%. Overall improvement is therefore
not an across-the-board neutral-recognition improvement. Raw SNLI NLL is 0.5990
versus 0.7408. The paired case-group bootstrap interval for the 7.8-point accuracy
difference is [0.5, 15.5] points; it excludes seed/development-selection uncertainty
and is not adjusted for multiple comparisons.

Intent is 59/61 versus 57/61; Schema is 21/60 for both; code strategy is 7/10
versus 8/10; emotion is 26/61 versus 25/61. Grouped emotion raw NLL (2.9259)
worsens relative to the initial model (2.7582) and dispersed (2.5558), despite
higher accuracy. Validation-only temperature scaling further worsens grouped
emotion NLL to 3.5878. Both checkpoints retain 25/25 order stability, with 13/25
versus 16/25 all-order correctness on that small separate audit subset.

This initial comparison used one seed; the replication is reported below.
Relative to historical studies, training examples,
update-level label balance and epochs also change. Only the within-study arms
isolate grouping, subject to the batching/dropout differences. No deployment
winner, general zero-shot competence or RL-estimator advantage is established.
See `docs/RELATION_GROUP_RESULTS.zh-CN.md` and the frozen design.
The unchanged authored evidence-flip probe scores 14/18 and 2/6 complete groups
for grouped, versus 15/18 and 4/6 for dispersed. Earlier model versions passed
zero complete groups. This diagnostic favors dispersed and is not a fresh
external benchmark; its inputs never enter training. Both evaluated weight
fingerprints match this study. See `docs/RELATION_GROUP_FLIPS.zh-CN.md`.

### Seed43 replication

The identical dataset, initial checkpoint, update budget and evaluation are run
again with seed43. The only config changes are the seed and its frozen epoch row
orders. Grouped reaches 115/192 (59.9%) versus dispersed 97/192 (50.5%), so the
overall grouped-minus-dispersed effect has the same direction as seed42: +9.4
points versus +7.8 points. The descriptive two-seed means are 68.0% versus 59.4%
for SNLI accuracy and 0.7420 versus 0.8799 for raw NLL.

Per-class behavior remains unstable. At seed43, grouped gets 57/64 contradiction,
42/64 neutral and only 16/64 entailment cases correct; dispersed gets 34/64,
34/64 and 29/64. Across the two seeds, grouped neutral recall averages 66.4%
versus 63.3%, and neutral precision averages 55.7% versus 50.5%. These two values
are descriptive averages, not seed-level confidence estimates.

Old-task outcomes also vary. Seed43 grouped versus dispersed is 57/61 vs 57/61
intent, 21/60 vs 16/60 Schema, 8/10 vs 5/10 code strategy, and 24/61 vs 20/61
emotion. All four seed/group checkpoints retain 25/25 option-order stability.
On the authored evidence-flip diagnostic, seed43 grouped scores 13/18 and 1/6
complete groups, versus dispersed 11/18 and 0/6. This reverses the diagnostic's
seed42 ordering, so no grouping winner is established there.

Two seeds make the overall grouping effect worth further study, but do not prove
a stable default, general zero-shot ability or an RLCD algorithmic advance. The
same development tests were already inspected, SNLI participates in training,
and most source rows have only one upstream annotation. See
`docs/RELATION_GROUP_MULTISEED.zh-CN.md` and its machine-readable evidence.

### Second-epoch ablation

For each seed, a new one-epoch endpoint is the exact prefix of the existing
balanced dispersed two-epoch run: the 512 executed microbatches, candidate
orders, first 128 optimizer records and first curve point match exactly. The
second epoch changes seed42 SNLI from 76/192 (39.6%) to 131/192 (68.2%), and
seed43 from 74/192 (38.5%) to 97/192 (50.5%). The descriptive mean change is
+20.3 points. It adds both 128 optimizer updates and a second exposure to every
selected row, so this study does not separate update count from repetition.

Per-class effects are uneven. Seed42 changes from 6/20/50 to 38/47/46 correct
contradiction/neutral/entailment cases; seed43 changes from 2/63/9 to 34/34/29.
The second epoch therefore improves total accuracy in both seeds while also
rebalancing which classes are learned. Old-task effects vary by task and seed.
See `docs/SECOND_EPOCH_RESULTS.zh-CN.md` and its machine-readable evidence.

### Update-level label-balance ablation

The balanced dispersed endpoints are compared with two new variable-label
endpoints. Dataset, two-epoch budget, source position, premise `group_id`, replay
ID and initial tensors are fixed. The treatment permutes each premise's three
relation rows over its three fixed positions, which allows per-update label
counts to vary while preserving one exposure per row per epoch. This also changes
which hypothesis text receives a particular microbatch/dropout assignment.

Balanced minus variable-label SNLI accuracy is +32.3 points at seed42
(131/192 vs 69/192) and +12.0 points at seed43 (97/192 vs 74/192), a descriptive
two-seed mean of +22.1 points. The variable seed42 endpoint collapses away from
neutral (0/64 correct), while variable seed43 collapses away from contradiction
(0/64 correct). Balanced per-class results are 38/47/46 and 34/34/29. On the
authored evidence-flip diagnostic, variable/balanced is 7/18 vs 15/18 at seed42
and 5/18 vs 11/18 at seed43; variable passes 0/6 complete groups in both seeds.

This supports update-level label balance for this training protocol, not a
general theorem. SNLI is a trained task, the evaluation has been repeatedly
inspected, and old-task results do not improve uniformly. See
`docs/LABEL_BALANCE_RESULTS.zh-CN.md`, `docs/LABEL_BALANCE_DESIGN.zh-CN.md` and
their machine-readable evidence.

### Frozen WinoGrande task-family transfer evaluation

Before any scores were observed, 384 WinoGrande 1.1 development cases were
selected by `SHA-256(qID, sentence, option1, option2)` with the answer excluded,
and nine checkpoints were predeclared: the common start plus one-epoch,
variable-label, balanced-dispersed and balanced-grouped endpoints for seeds
42/43. WinoGrande examples are absent from project fine-tuning and calibration.

Balanced minus variable-label accuracy is +0.3 points at seed42 (200/384 versus
199/384) and -0.3 at seed43 (195/384 versus 196/384). Balanced minus the common
start is +1.6 and +0.3 points; both paired case-bootstrap intervals cross zero.
All nine checkpoints score between 49.0% and 52.1%. Balanced raw NLL is worse
than variable-label NLL by 0.2818 and 0.0883 at the two seeds. Grouping changes
accuracy by -3.1 and -0.3 points versus balanced dispersed checkpoints.

Every checkpoint predicts the content in the upstream `option1` field much more
often (65.4%–77.9%) than its 50.3% true frequency. Independent candidate encoding
is permutation equivariant and never encodes choice IDs, so this indicates a
content/distribution preference rather than a fixed output-index head. The
experiment does not show cross-task commonsense transfer from the relation
training gains. It also cannot rule out WinoGrande contamination in the public
base model's pretraining corpus. See `docs/WINOGRANDE_BLIND_RESULTS.zh-CN.md`.

The official WinoGrande repository states that the dataset is CC-BY but does not
identify a version. Raw and transformed WinoGrande text is therefore kept out of
release bundles; the project distributes download/checksum/selection code and
aggregate evidence only.

### Subsequent PAWS task-expansion study

Two final checkpoints warm-start the seed42/43 balanced dispersed relation
checkpoints and train once over 768 PAWS-Wiki Labeled (Final) rows plus the same
256 old-task replay rows. Each run uses clean CE + 0.5 Brier, 128 updates and a
fixed final endpoint. PAWS test accuracy changes from 98/192 to 143/192 at
seed42 (+23.4 points) and 96/192 to 101/192 at seed43 (+2.6 points). SNLI changes
from 131/192 to 123/192 and 97/192 to 81/192. Every update in both runs triggers
gradient-norm clipping.

Before training, 384 token-admissible BoolQ development cases are frozen by
`SHA-256(question, passage)` without using answers. Both training endpoints finish
before any BoolQ inference. BoolQ changes from 237/384 to 247/384 at seed42 and
238/384 to 185/384 at seed43. The starts predict yes on 384 and 375 cases; the
finals predict yes on 356 and 124 cases. WinoGrande changes from 200/384 to
197/384 and 195/384 to 194/384. These results show large candidate-preference
movement and do not establish stable cross-task transfer. See
`docs/TASK_EXPANSION_RESULTS.zh-CN.md`.

The study lacks a matched-compute non-PAWS continuation control, uses two seeds
with different relation-trained starting weights, and cannot exclude public-base
pretraining contamination. PAWS, BoolQ and WinoGrande text is excluded from
release bundles; pinned source/license snapshots and aggregate evidence remain.

### Common-start task-and-label stratification study

Four fixed endpoints share the seed42 balanced-relation checkpoint and the same
384 PAWS, 384 SNLI and 256 old-task replay rows. Each row appears once; each arm
uses 128 updates, clean CE + 0.5 Brier and a fresh optimizer. The treatment fixes
every update to 3 PAWS, 3 SNLI and 2 replay rows, with one example per SNLI label
and globally balanced PAWS labels. The control randomly mixes the same rows.

Stratified minus mixed accuracy is +1.0/+2.1 points on PAWS and +2.6/+17.7
points on SNLI at seeds 42/43. PAWS raw NLL becomes 0.0251/0.0108 worse, while
SNLI raw NLL improves by 0.1344/0.1978. CLINC and Schema point estimates improve
in both seeds; the 10-case code-policy subset loses 1 and 3 correct cases.

The previously inspected BoolQ diagnostic changes by -1.6/+4.4 points and
WinoGrande by -2.3/+1.6 points for stratified versus mixed, so cross-task transfer
does not replicate. Candidate preferences remain unstable: BoolQ `yes` predictions
are 292 versus 310 at seed42 and 156 versus 339 at seed43.

At the common start, eight no-dropout task-gradient probes find positive PAWS/SNLI
cosine in all eight samples (mean 0.130), while PAWS/replay and SNLI/replay are
negative in five of eight samples. Stratification does not control gradient
magnitude: maximum pre-clip norm changes from 111.7 to 8929.9 at seed42 and 51.2
to 604.4 at seed43. Every update in every arm is clipped. Shared seeds do not
make candidate permutations, padding work or dropout masks identical because
task shapes and row order change RNG consumption. See
`docs/TASK_BALANCE_RESULTS.zh-CN.md`.

### Common-start task-gradient control study

Six endpoints reuse the stratified rows and exact within-seed update plans from
the task-balance study. Each update computes separate PAWS, SNLI and replay
gradients. The treatments either cap task norms at the within-update median
without amplifying smaller gradients, or apply a deterministic symmetric
projection to task pairs with negative dot products. All arms then use the same
row-share weighting, global norm clipping and fresh optimizer.

Median capping reduces p95 pre-clip norms from 57.53/32.74 to 29.67/29.25 at
seeds 42/43, but PAWS accuracy changes by -3.6/-0.5 points. Symmetric projection
changes SNLI by +1.6/+1.0 and overall hard accuracy by +0.6/+1.4 points, while
PAWS changes by +0.5/-3.1; its maximum norms remain 154.59/371.32. Every paired
PAWS/SNLI group-bootstrap interval crosses zero. BoolQ and WinoGrande were
already inspected and are diagnostics only. Neither treatment is a promoted
default. See `docs/TASK_GRADIENT_CONTROL_RESULTS.zh-CN.md`.

### Soft-cap then projection follow-up

A follow-up reuses both raw controls bit-for-bit and trains four new endpoints.
A 2x within-update median cap followed by symmetric conflict projection changes
PAWS by +0.5/-0.5 and SNLI by +4.7/+4.2 points at seeds 42/43. All four paired
group-bootstrap intervals cross zero. Pre-clip p95 falls from 57.53 to 31.41 at
seed42 but rises from 32.74 to 37.50 at seed43, so the preregistered advancement
rule fails. This treatment remains a research candidate rather than a promoted
default. See `docs/TASK_GRADIENT_COMPOSITION_RESULTS.zh-CN.md`.

### Post-projection trust-region result

The registered trust-region follow-up adds a deterministic bound after soft
capping and conflict projection. All 384 trust updates across seeds 42/43/44
obey the bound, with a maximum numerical excess of 5.41e-08. This verifies the
implementation's geometric invariant. It does not verify a quality benefit:
mean PAWS/SNLI accuracy changes by -1.0 percentage point versus raw, the worst
task/seed change is -7.8 points, and gradient p95 does not improve consistently.
The method is not a default and was not advanced to a fresh blind task. See
`docs/TASK_GRADIENT_TRUST_RESULTS.zh-CN.md`.

### Encoder/decoder route decision

The frozen three-seed comparison is complete. All six endpoints use identical
public rows, within-seed request plans, update counts, LoRA layer/rank settings
and objectives. The independent validator confirms that MiniCPM5 passes all
eight preregistered advancement conditions across trained tasks, retained and
task-held-out tasks, order audits and sealed ARC. It is therefore the selected
route for the next research stage. ARC total accuracy is 81.2%/81.8%/81.2%
across seeds versus 35.7%/35.4%/33.1% for EuroBERT; local median inference is
about 10%–11% slower. See `docs/ENCODER_DECODER_RESULTS.zh-CN.md` and the
machine-readable evidence. This compares two particular pretrained bases and
does not isolate attention direction or prove universal zero-shot competence.

### Candidate-prior correction status

The frozen three-seed prior-correction study selects `lambda=0` from seeds 42/43
validation NLL. Equal-seed NLL worsens monotonically from 0.5851 without
subtraction to 0.6773 at full subtraction. The method therefore does not enter
the default inference path. After selection was hashed, all three endpoints were
evaluated once on 288 answer-blind-selected BIG-bench logical-deduction cases.
Across seeds, accuracy is 78.1%–79.2% for three choices, 46.9%–50.0% for five,
and 65.6%–67.7% for seven. Every 12-case order and candidate-ID audit is stable.
See `docs/CANDIDATE_PRIOR_RESULTS.zh-CN.md`. Logical deduction is now an observed
diagnostic and cannot serve as fresh evidence for subsequent method tuning.

Most fine-tuning data is English. Other languages, long contexts, adversarial
inputs, out-of-scope questions and many-choice decisions have not been validated.
Probabilities may be overconfident, especially on unseen tasks. The interface
keeps `requires_review: true`; no automatic-action threshold has been established.

## Reuse

Use the project's `predict` command to load the public base plus checkpoint.
Use `train --init-from` for further fine-tuning with a new optimizer and a new
validation calibration. The project code is Apache-2.0; underlying data and base
model terms are documented separately. Keep source attribution when sharing
derived datasets and training artifacts.
