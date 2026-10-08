# Third-party attribution

Project code is Apache-2.0. Dataset licenses apply separately to source and
derived data. The downloader and transformations do not remove attribution or
relicense upstream material. No endorsement by the following authors is implied.

| Material | Attribution and source | License |
|---|---|---|
| EuroBERT-2.1B original weights | [EuroBERT authors, Boizard et al. (2025)](https://huggingface.co/EuroBERT/EuroBERT-2.1B) | Apache-2.0 |
| MiniCPM5-2B-Base original weights | [OpenBMB / ModelBest](https://huggingface.co/openbmb/MiniCPM5-2B-Base) | Apache-2.0 in the official model metadata; revision pinned locally, base weights excluded from release bundles |
| JSON Schema Test Suite | [Julian Berman and contributors](https://github.com/json-schema-org/JSON-Schema-Test-Suite) | MIT; retained at `third_party/json-schema-LICENSE` |
| Bandit examples | [PyCQA / OpenStack Bandit contributors](https://github.com/PyCQA/bandit) | Apache-2.0; retained at `third_party/bandit-LICENSE` |
| CLINC150 | [Stefan Larson et al. (2019), An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://github.com/clinc/oos-eval) | CC-BY-3.0; retained at `third_party/clinc-LICENSE` |
| GoEmotions | [Dorottya Demszky et al. (2020), GoEmotions: A Dataset of Fine-Grained Emotions](https://github.com/google-research/google-research/tree/master/goemotions) | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); data license stated in the pinned Google Research root README |
| SNLI 1.0 | [Stanford NLP / Bowman et al. (2015), SNLI](https://nlp.stanford.edu/projects/snli/) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); exact archive hash, member files and derived row provenance are pinned |
| Typed Decisions | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Apache-2.0 in pinned dataset metadata; every derived row records revision, workflow, case and teacher-label semantics |
| WinoGrande 1.1 blind evaluation | [Keisuke Sakaguchi et al. (2019), WinoGrande](https://github.com/allenai/winogrande) | Dataset CC-BY, version unspecified in the official README; notice retained at `third_party/winogrande-DATA-NOTICE`; raw text excluded from release bundles |
| PAWS-Wiki Labeled (Final) | [Google Research, Zhang et al. (2019), PAWS](https://github.com/google-research-datasets/paws) | Official data permission allows use for any purpose and requests acknowledgement; retained at `third_party/paws-DATA-LICENSE` |
| BoolQ frozen evaluation | [Google Research, Clark et al. (2019), BoolQ](https://github.com/google-research-datasets/boolean-questions) | CC BY-SA 3.0 as stated in the official README; retained at `third_party/boolq-UPSTREAM-README` |
| QASC evidence-composition training | [Allen Institute for AI, Khot et al. (2019), QASC](https://github.com/allenai/qasc) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), explicitly stated by the official repository and pinned dataset metadata; notice retained at `third_party/qasc-UPSTREAM-README` |
| AI2 Reasoning Challenge frozen evaluation source | [Allen Institute for AI, Clark et al. (2018), ARC](https://huggingface.co/datasets/allenai/ai2_arc) | CC BY-SA 4.0 in the official dataset metadata; source revision and files pinned, raw/transformed rows excluded from release bundles |
| BIG-bench logical_deduction frozen evaluation | [Google BIG-bench contributors](https://github.com/google/BIG-bench/tree/092b196c1f8f14a54bbc62f24759d43bde46dd3b/bigbench/benchmark_tasks/logical_deduction) | Apache-2.0; retained at `third_party/bigbench-LICENSE`; repository revision and five source files pinned, raw/transformed rows excluded from release bundles |
| BIG-bench release-candidate blind suite | [Google BIG-bench contributors](https://github.com/google/BIG-bench/tree/092b196c1f8f14a54bbc62f24759d43bde46dd3b/bigbench/benchmark_tasks) | Apache-2.0; `causal_judgment`, `formal_fallacies_syllogisms_negation` and `movie_recommendation`; same retained license and pinned revision; raw/transformed rows excluded from release bundles |
| Mostly Basic Python Problems outcome training | [Google Research, Austin et al. (2021)](https://github.com/google-research/google-research/tree/f46ca8374b4cddef97ca4208ad986049d74d296a/mbpp) | CC-BY-4.0; source revision/file hash pinned in `configs/mbpp-outcomes-source.json`, split and data-license notices retained at `third_party/mbpp-UPSTREAM-README` and `third_party/google-research-DATA-LICENSE-NOTICE` |
| TruthfulQA | [Lin et al., official repository](https://github.com/sylinrl/TruthfulQA) | Apache-2.0; revision and source hashes pinned, raw/transformed rows excluded from release bundles |
| HellaSwag | [Zellers et al. (2019), official repository](https://github.com/rowanz/hellaswag) | MIT; pinned Hugging Face conversion and file hashes, raw/transformed rows excluded from release bundles |
| GSM8K | [Cobbe et al. (2021), official repository](https://github.com/openai/grade-school-math) | MIT; pinned Hugging Face conversion and file hashes, raw/transformed rows excluded from release bundles |
| SciFact sealed evaluation | [Wadden et al. (2020), official repository](https://github.com/allenai/scifact) | Claims/evidence annotations CC-BY-4.0; abstract corpus ODC-By-1.0; code Apache-2.0; opened once after the final protocol, seeds, weights and gates were frozen; raw/transformed text remains excluded |

The Google Research repository distinguishes **datasets (CC-BY-4.0)** from code
(Apache-2.0); we use the data license for GoEmotions, not the root code license.
CLINC150 comes from its original author repository under CC-BY-3.0.

MBPP is used to derive executable candidate-outcome tables. The preparation
keeps the upstream task attribution, normalizes the reference program, generates
deterministic single-AST-edit mutants, and runs the upstream tests in isolated
subprocesses. Development excludes official test IDs 11–510. Raw and transformed
rows remain in ignored local directories; source/config hashes and aggregate
evidence can be included in a release. See
`docs/EXECUTABLE_OUTCOME_RLCD_DESIGN.zh-CN.md`.

Modifications: source examples are converted to task/context/choices records;
some examples are filtered, descriptions normalized, candidate sets sampled,
and bounded-ambiguity examples derived. Bandit code is reduced to imports and
call expressions and labeled by explicit narrow policies. See `docs/DATA.md`
and each row's provenance. Original source commits and file hashes appear in
`sources.lock.json`.

The public EuroBERT base is loaded directly from its pinned original revision.
Fresh adapter/head training does not inherit weights from prior private-data
experiments. The exact pretraining corpus of the base is outside this project's
control; "public-data-only" describes our fine-tuning pipeline.

The decoder comparison uses the public MiniCPM5-2B-Base revision
`96a57cd572a02506b4500f54427dca24970c1bac`. Its local fetcher records hashes
for the original configuration, tokenizer, README and weights. The project does
not redistribute those base files; users fetch and verify the pinned upstream
checkpoint before running the comparison.

The optional causal shared-prefix inference path uses a common-prefix cache and
keeps its own independently scored candidate head and API. It does not include
third-party weights, training data or benchmark claims.

## Typed Decisions soft-target data

`LocalLLaMA/typed-decisions` is pinned at revision
`ea9306458d6e9563628369a3d1e72e362fb381d2` under Apache-2.0. The preparation
script verifies the upstream README and combined train/test Parquet files by
size and SHA-256. Prepared rows retain state, question/candidate semantics and
public teacher probability targets; latent generation factors and diagnostics
are excluded. The dataset measures teacher agreement rather than verified action
success. Raw and prepared files remain in ignored local directories and are not
relicensed by this project. See `configs/typed-decisions-source.json` and
`docs/TYPED_DECISIONS_STUDY.zh-CN.md`.

## Additional SNLI data and continuation training

`scripts/prepare_snli.py` separately prepares an SNLI subset. SNLI is **not** part
of any initial-pilot, calibration-study-v1 or architecture-study-v1 training data.
Source: [Stanford NLP Group, Bowman et al. (2015)](https://nlp.stanford.edu/projects/snli/),
*A large annotated corpus for learning natural language inference*, under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The output retains upstream README and attribution; transformations include
task/candidate formatting, normalized duplicate/conflict and cross-split premise
filtering, and deterministic label-balanced selection. See
`configs/snli-source.json` and `docs/SNLI_PREPARATION.md`.

The subsequent `gradient-training-v1` checkpoints each use the same 48
SNLI-derived training records with 144 old-task replay records. The combined
dataset preserves individual source licenses and provenance. Adapter bundles
contain model/evidence files, not the raw or prepared datasets.

## WinoGrande blind evaluation

WinoGrande 1.1 is used only for a locally frozen, zero-shot transfer evaluation;
it does not enter training or calibration. `scripts/prepare_winogrande_blind.py`
downloads the official archive, verifies SHA-256, and selects cases without using
answers. Reports and machine-readable aggregate evidence can be distributed,
while raw or transformed WinoGrande text stays outside release bundles. The
official repository describes the dataset license only as CC-BY and does not
state a version; downstream users should review the upstream terms directly.

## PAWS-Wiki training and BoolQ frozen evaluation

The task-expansion study uses only PAWS-Wiki Labeled (Final), not PAWS-QQP.
Parquet conversions are pinned to Hugging Face dataset revision
`161ece9501cf0a11f3e48bd356eaa82de46d6a09`; every file has a local SHA-256.
The upstream PAWS license at archived commit
`02b29f3af1143620d1b7f352247e65c0bdd2ec18` explicitly permits use of the
dataset for any purpose and asks for acknowledgement. Training uses 768
label-balanced official-train rows and keeps separate official validation/test
subsets. The transformed data is not relicensed as Apache-2.0.

BoolQ is evaluation-only. Its official README at archived commit
`90af34107399cc7a446b373dc4ee35b8001da7c2` states CC BY-SA 3.0. The local
parquet conversion is pinned to `google/boolq` revision
`35b264d03638db9f4ce671b711558bf7ff0f80d5`. The 384 evaluation rows are chosen
by a hash of question and passage after input-length admission; the answer is
excluded from selection. Release bundles keep source code, checksums and
aggregate evidence, while transformed PAWS/BoolQ text remains outside bundles.

## QASC evidence-composition training

The QASC screen uses the official train and validation splits through the pinned
`allenai/qasc` conversion revision
`a34ba204eb9a33b919c10cc08f4f1c8dae5ec070`. The official QASC repository says
the data is released under CC BY; the pinned dataset metadata specifies
CC-BY-4.0. Preparation verifies the README and both Parquet files by SHA-256.

Each selected row keeps the question, `fact1`, `fact2`, and all eight candidate
texts. It excludes `combinedfact`, hides the source A–H letters, and orders
candidates by an answer-blind hash of question and candidate text. Training and
validation each contain equal counts for all eight resulting correct positions.
The 128 validation rows become consumed development data after this screen.
Raw and transformed QASC text remains in ignored local directories; release
materials contain code, source pins, attribution, and aggregate evidence only.

## ARC sealed task-level evaluation

ARC was used once as the formal architecture comparison's task-level blind test;
it did not enter training, calibration, early stopping or endpoint selection.
The official revision `210d026faf9955653af8916fad021475a3f00453` and two
validation parquet files are pinned in `configs/arc-blind-source.json`. After
all six endpoints and engineering audits were fixed, the preparation script
selected 192 ARC-Easy and 192 ARC-Challenge cases by an answer-blind input hash
and both tokenizers' length checks. Raw rows, transformed cases and per-case
predictions stay outside release bundles; aggregate metrics and protocol hashes
may be published.

## BIG-bench logical-deduction blind evaluation

The candidate-prior study pins the archived official BIG-bench repository at
`092b196c1f8f14a54bbc62f24759d43bde46dd3b` under Apache-2.0. It uses only the
`logical_deduction` README and 3/5/7-object JSON tasks. For each subtask, 96 rows
are selected by a hash of subtask, input and ordered choice text; target scores
are excluded from selection. The resulting 288 rows are evaluated only after a
shared prior-correction strength is selected and hashed from seed42/43 validation
data. Raw/transformed rows, logits and per-case predictions remain outside release
bundles; source hashes and aggregate evidence may be published.

## BIG-bench release-candidate blind suite

The release study uses the same pinned BIG-bench repository revision and license.
It verifies each selected task's README and `task.json`, then admits examples
under the pinned decoder tokenizer. Selection sorts by a hash of task, input and
ordered candidate text; target-score values do not enter the key. It retains 128
rows from each of `causal_judgment`, `formal_fallacies_syllogisms_negation` and
`movie_recommendation`.

The final case file may be materialized only after all six matched endpoint
weights and engineering audits have been checksummed. It is then used once for
the frozen advancement decision and never for training, temperature fitting,
early stopping, endpoint selection or hyperparameter selection. Raw/transformed
rows, labels, logits and per-case predictions remain outside release bundles.
