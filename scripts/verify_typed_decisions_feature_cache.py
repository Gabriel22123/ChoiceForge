"""Verify cached float16 candidate states against serial base-model forwards."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from decision_model.core import digest, load_rows, write_json
from decision_model.decoder_model import DecoderJudge, verify_local_decoder_base
from decision_model.feature_screen import CandidateHead, FeatureStore, padded_logits


THRESHOLDS = {
    "max_relative_l2_error": 0.0005,
    "min_cosine_similarity": 0.999999,
    "max_logit_abs_error": 0.005,
    "max_probability_abs_error": 0.001,
    "required_argmax_matches": 16,
}


def selected_rows(fit_root, count=16):
    unique = {}
    for path in sorted(Path(fit_root).glob("heldout-*/fit.jsonl")):
        for row in load_rows(path):
            unique.setdefault(row["id"], row)
    ranked = sorted(unique.values(), key=lambda row: hashlib.sha256(
        ("typed-decisions-feature-cache-parity-v1:" + row["id"]).encode()).hexdigest())
    return ranked[:count]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="cache/typed-decisions-minicpm5-features-v1")
    parser.add_argument("--fit-root", default="data/typed-decisions-objective-screen-v1")
    parser.add_argument("--base-path", default="models/minicpm5-2b-base")
    parser.add_argument("--output", default="docs/evidence/typed-decisions-feature-cache-parity-v1.json")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    store = FeatureStore(root / args.cache)
    frozen = store.manifest["frozen"]
    provenance_checks = {
        "extractor_source": frozen["extractor_source_sha256"] == digest(
            (root / "scripts/cache_typed_decisions_decoder_features.py").read_bytes()),
        "decoder_source": frozen["decoder_model_source_sha256"] == digest(
            (root / "src/decision_model/decoder_model.py").read_bytes()),
        "base_lock": frozen["base_lock_sha256"] == digest(
            (root / "src/decision_model/decoder_base.lock.json").read_bytes()),
        "fold_sources": all(digest((root / path).read_bytes()) == expected
                            for path, expected in frozen["source_sha256"].items()),
    }
    if not all(provenance_checks.values()):
        raise ValueError("Feature-cache source provenance is stale")
    rows = selected_rows(root / args.fit_root)
    if len(rows) != THRESHOLDS["required_argmax_matches"]:
        raise ValueError("Parity sample count differs from frozen threshold")
    features = store.load_subset([row["id"] for row in rows])
    verify_local_decoder_base(root / args.base_path)
    config = dict(frozen["config"])
    judge = DecoderJudge(config, base_path=root / args.base_path, device=args.device); judge.train(False)
    if not hasattr(judge.encoder, "disable_adapter"):
        raise ValueError("Formal parity requires an explicit adapter-disabled context")
    import torch
    torch.manual_seed(20260926)
    head = CandidateHead.build(store.manifest["hidden_size"], 256, 0.0).eval()
    direct_rows = []; cached_rows = []; details = []
    with torch.no_grad(), judge.encoder.disable_adapter():
        for row in rows:
            direct = judge.features([judge.encode(row["request"])])[0, :len(row["request"]["choices"])].cpu()
            cached = features[row["id"]].float(); direct_rows.append(direct); cached_rows.append(cached)
            difference = (direct - cached).float()
            relative = float(difference.norm() / direct.float().norm().clamp_min(1e-12))
            cosine = float(torch.nn.functional.cosine_similarity(
                direct.float().flatten(), cached.float().flatten(), dim=0))
            details.append({"id": row["id"], "candidates": len(cached),
                            "max_abs_error": float(difference.abs().max()),
                            "mean_abs_error": float(difference.abs().mean()),
                            "relative_l2_error": relative, "cosine_similarity": cosine})
    with torch.no_grad():
        direct_logits, _ = padded_logits(head, direct_rows)
        cached_logits, _ = padded_logits(head, cached_rows)
    counts = [len(row) for row in direct_rows]
    probability_error = []; argmax_matches = 0
    for index, count in enumerate(counts):
        left = direct_logits[index, :count].softmax(-1); right = cached_logits[index, :count].softmax(-1)
        probability_error.append(float((left - right).abs().max()))
        argmax_matches += int(left.argmax().item() == right.argmax().item())
    summary = {
        "rows": len(rows), "candidate_paths": sum(counts),
        "max_relative_l2_error": max(value["relative_l2_error"] for value in details),
        "min_cosine_similarity": min(value["cosine_similarity"] for value in details),
        "max_feature_abs_error": max(value["max_abs_error"] for value in details),
        "max_logit_abs_error": float((direct_logits - cached_logits).abs().max()),
        "max_probability_abs_error": max(probability_error), "argmax_matches": argmax_matches,
    }
    checks = {
        "provenance": all(provenance_checks.values()),
        "relative_l2": summary["max_relative_l2_error"] <= THRESHOLDS["max_relative_l2_error"],
        "cosine": summary["min_cosine_similarity"] >= THRESHOLDS["min_cosine_similarity"],
        "logits": summary["max_logit_abs_error"] <= THRESHOLDS["max_logit_abs_error"],
        "probabilities": summary["max_probability_abs_error"] <= THRESHOLDS["max_probability_abs_error"],
        "argmax": summary["argmax_matches"] == THRESHOLDS["required_argmax_matches"],
    }
    evidence = {"status": "pass" if all(checks.values()) else "fail",
                "selection": "16 fit-only row IDs ranked by fixed SHA-256; labels and probabilities excluded from ranking",
                "cache_manifest_sha256": store.manifest_sha256,
                "script_sha256": digest(Path(__file__).read_bytes()),
                "cache_provenance_checks": provenance_checks,
                "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                            "device": args.device, **{package: importlib.metadata.version(package)
                            for package in ("torch", "transformers", "peft", "safetensors")}},
                "thresholds_frozen_in_source_before_measurement": THRESHOLDS,
                "summary": summary, "checks": checks, "rows_detail": details}
    write_json(root / args.output, evidence); print(json.dumps(evidence, indent=2))
    if evidence["status"] != "pass":
        raise SystemExit("Feature-cache parity gate failed")


if __name__ == "__main__":
    main()
