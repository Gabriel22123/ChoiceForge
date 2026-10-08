# Release bundle quickstart

This guide applies only when `docs/evidence/release-candidate-v1.json` says
`"eligible": true` and a bundle has been produced by
`scripts/package_release_candidate.py`. A failed release study does not produce
or authorize a checkpoint bundle.

## 1. Install

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[model,data]'
```

The adapter does not contain the 2B base model. On first use, Transformers
downloads the exact pinned `openbmb/MiniCPM5-2B-Base` revision. To stay offline,
download and verify the original base separately, then pass
`--base-path /path/to/minicpm5-2b-base`.

## 2. Make a bounded decision

```bash
decision-model predict \
  --checkpoint checkpoint \
  --input examples/request.json
```

Typed Boolean and ordered Score requests use the same command:

```bash
decision-model predict --checkpoint checkpoint --input examples/boolean-request.json
decision-model predict --checkpoint checkpoint --input examples/score-request.json
```

The runtime scores candidates without text generation. It constructs JSON in
code, validates the exact field set and probability invariants, and always emits
`"requires_review": true`. It cannot emit chain-of-thought or arbitrary JSON
fields through this interface.

## 3. Reproduce the canonical adapter training

The bundle includes the exact 5,056-row Expanded JSONL used by the frozen study.
It contains 3,072 public training rows, 608 validation rows and 1,376 common test
rows. Dataset licenses remain upstream-specific; see `THIRD_PARTY.md`.

```bash
decision-model train \
  --data data/release-candidate-v1/expanded.jsonl \
  --config checkpoint/model.json \
  --output runs/reproduced-seed1701-expanded
```

Add `--base-path` and `--device cpu|mps|cuda` as needed. The final epoch is fixed;
test results do not select a checkpoint. Exact floating-point equality across
different accelerators is not promised. The run records versions, selected row
IDs, update plan, initialization hash and artifact checksums so differences can
be audited.

## 4. Continue training on your own tasks

Create JSONL rows following `docs/DATA.md`. Keep whole source groups in one split
and hold out entire task families when measuring transfer. Then start a new
optimizer from the released adapter/head:

```bash
decision-model train \
  --data path/to/your-cases.jsonl \
  --config checkpoint/model.json \
  --init-from checkpoint \
  --output runs/my-adaptation
```

Do not overwrite the original checkpoint. A downstream adapter needs its own
model card, data provenance, evaluation and calibration evidence.

## 5. Verify the bundle

`release-manifest.json` lists the SHA-256 and byte count of every included file.
The bundle excludes base weights, blind-suite cases, labels, per-case blind
predictions, logits, caches and all private files. The aggregate release report
and hashes remain available for independent review.

```bash
python scripts/verify_release_bundle.py .
```

The verifier checks every declared file, rejects symlinks and forbidden canary
paths, permits only the adapter/head safetensors file, and requires a successful
13-part project-completion audit in the final bundle.
