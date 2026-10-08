# Optional public SNLI preparation

This is a separately prepared dataset. It is **not** used by the initial
pilot, calibration study, or `architecture-study-v1` checkpoints.
The later `gradient-training-v1` study uses 48 training records from this subset;
see [the matched model study](GRADIENT_TRAINING_STUDY.zh-CN.md).
The later [semantic expansion](SEMANTIC_SCALE_STUDY.zh-CN.md) uses 768 training
records per seed, with the same subset repeated across two independent runs.

The local preparation produced 4,224 records (3,072 train / 384 validation /
768 test), with byte-identical reconstruction checked. Data SHA-256:
`d2e11f0e7b83a5a89c72d79b637cde9c6f5673fc398e4c6850b3063e11e75aec`.
Filtering removed 1,119 unknown/no-consensus-label rows, 218 cross-split-premise
rows, 151 conflicting-label rows and 570 duplicate rows, before bounded sampling.
These are sequential exclusion counts, not overlapping totals. See
[the preparation record](evidence/public-snli-preparation-v1.json).

## Reproduce

Download the official archive from the [SNLI project page](https://nlp.stanford.edu/projects/snli/)
and keep it locally. The exact archive SHA-256 and URL are in
`configs/snli-source.json`. Then run:

```bash
python scripts/prepare_snli.py \
  --archive /path/to/snli_1.0.zip \
  --output data/public-snli-v1
```

The default caps are 1,024 train, 128 validation and 256 test examples **per
label**. There are three labels, so the maximum output is 3,072/384/768 rows.
The actual counts are recorded in `manifest.json`. A new output directory is
required. No model training occurs in this command.

The script verifies the archive before parsing, uses the official JSONL splits,
and performs two streaming passes with compact fingerprints. It does not extract
arbitrary archive paths. Output includes `cases.jsonl`, `manifest.json`, attribution
and the unchanged upstream README. The existing `decision-model validate-data`
contract is applied to the prepared cases.

## Statistical and semantic choices

- Remove labels outside entailment/contradiction/neutral, including no-consensus
  examples. Keep upstream gold labels, without teacher-generated replacements.
- Normalize strings with Unicode NFKC, case folding and whitespace collapse for
  duplicate detection only. Preserve original text in the model input.
- Drop **all** rows belonging to normalized premises shared by official splits.
  This does not silently move test content into training.
- Drop normalized premise/hypothesis pairs with conflicting labels; deduplicate
  repeated ordered pairs. Semantic paraphrase leakage may still remain.
- Select the lowest content hashes per split and label. This is reproducible and
  label-balanced, not the original SNLI benchmark or a natural production prior.
- Candidate descriptions express support, contradiction or undetermined status
  separately. IDs, gold label, provenance and split are absent from model text.
- The task is seen-task evaluation if SNLI training examples are used. A new NLI
  dataset is not automatically a new task family.

Training on the entire prepared subset still needs a separately frozen model,
budget and evaluation plan. This preparation alone supplies no model result.

## Attribution and terms

The [official source](https://nlp.stanford.edu/projects/snli/) licenses SNLI under
CC BY-SA 4.0 and credits underlying caption sources. The derived subset retains
that data license and records its transformations; the project code's Apache-2.0
license does not replace the data terms. Do not bundle this data as unrestricted
Apache-licensed material. Keep the generated attribution and upstream README
alongside any shared derived dataset.

Reference: Samuel R. Bowman, Gabor Angeli, Christopher Potts and Christopher D.
Manning (2015), *A large annotated corpus for learning natural language inference*.
