# ChoiceForge workflow reference

Run commands from the repository root after installing the package with
`python -m pip install -e '.[data,model]'`.

## Canonical request

```json
{
  "task": "Choose the candidate that best satisfies the stated task.",
  "context": "Optional evidence or material.",
  "choices": [
    {"id": "a", "description": "First candidate"},
    {"id": "b", "description": "Second candidate"}
  ]
}
```

The input may also use the typed Boolean, Choice, or Score schemas described in
`docs/OUTPUT_CONTRACT.zh-CN.md`. IDs are caller-owned and may change per request.

## Use the released bundle

Download the pinned public MiniCPM base, then run:

```bash
python scripts/fetch_decoder_base.py --output models/minicpm5-2b-base
choiceforge predict-merged \
  --bundle models/choiceforge-local-bundle-v1 \
  --base-path models/minicpm5-2b-base \
  --input /path/to/request.json \
  --device auto
```

The runtime returns a fixed schema with `choice_id`, `probabilities`,
`requires_review`, `score_kind`, and `calibration`. It does not generate a
chain-of-thought field or arbitrary keys.

## Train a public-data adapter

```bash
choiceforge train \
  --data data/release-candidate-v1/expanded.jsonl \
  --config configs/eurobert-public-pilot.json \
  --output runs/my-public-adapter \
  --device auto
```

For a continuation, add `--init-from path/to/adapter` and keep the output path
separate. The release data audit is:

```bash
PYTHONPATH=src python scripts/audit_release_data.py
```

## Evaluate and audit

```bash
choiceforge evaluate \
  --checkpoint runs/my-public-adapter \
  --data data/release-candidate-v1/expanded.jsonl \
  --split validation \
  --output runs/my-public-evaluation \
  --device auto

choiceforge audit \
  --checkpoint runs/my-public-adapter \
  --data data/release-candidate-v1/expanded.jsonl \
  --output runs/my-public-audit \
  --device auto
```

Use validation for development and reserve test for the final locked report.
Keep the generated JSON evidence with the run; it records the input and model
hashes needed to reproduce the result.

## Public-data boundary

Only use datasets whose source, license, revision and transformation are
documented. Do not place private prompts, private logs, credentials, local
absolute paths or proprietary model files in a run that will be published.
