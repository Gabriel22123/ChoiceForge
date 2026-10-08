"""Portable inference runtime for a verified merged residual adapter bundle."""
from __future__ import annotations

import json
from pathlib import Path

from .core import digest, validate_request
from .expert_portfolio import decision_stable_projection
from .merged_adapter import build_merged_residual_adapter
from .output_contract import validate_prediction
from .prior_correction import prior_only_request


class MergedDecisionModel:
    def __init__(self, parent, adapter, metadata):
        if metadata.get("format") != "choiceforge-merged-residual-v1":
            raise ValueError("Unknown merged adapter format")
        architecture = metadata.get("architecture", {})
        if (architecture.get("hidden_size") is None or
                architecture.get("total_width") is None or
                not metadata.get("projection_alphas")):
            raise ValueError("Merged adapter metadata is incomplete")
        self.parent = parent
        self.adapter = adapter.to(parent.device)
        self.adapter.eval()
        self.metadata = metadata

    def predict(self, request):
        request = validate_request(request)
        torch = self.parent.torch
        self.parent.train(False)
        with torch.no_grad():
            encoded = [self.parent.encode(request),
                       self.parent.encode(prior_only_request(request))]
            features = self.parent.features(encoded)
            parent = self.parent.head(features).squeeze(-1)
            portfolio = parent + self.adapter(features).squeeze(-1)
            selected, _, _ = decision_stable_projection(
                parent[0], parent[1], portfolio[0], portfolio[1],
                self.metadata["projection_alphas"])
            probabilities = selected.softmax(-1).cpu().tolist()
        ids = [choice["id"] for choice in request["choices"]]
        prediction = {
            "choice_id": ids[max(range(len(ids)), key=probabilities.__getitem__)],
            "probabilities": dict(zip(ids, probabilities)),
            "requires_review": True,
            "score_kind": "candidate_probability_not_a_guarantee",
            "calibration": "uncalibrated",
        }
        return validate_prediction(prediction, request)


def load_merged_bundle(bundle_path, *, base_path, device="auto"):
    """Load and verify a local ChoiceForge bundle plus its separate public base."""
    from safetensors.torch import load_file
    from .judge_factory import make_judge, verify_base_for_config

    bundle = Path(bundle_path)
    manifest_path = bundle / "bundle.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("format") != "choiceforge-local-bundle-v1" or
            set(manifest.get("files", {})) != {
                "adapter.json", "adapter.safetensors", "parent/model.json",
                "parent/decision.safetensors", "parent/calibration.json"}):
        raise ValueError("ChoiceForge bundle manifest is invalid")
    for relative, expected in manifest["files"].items():
        if digest((bundle / relative).read_bytes()) != expected:
            raise ValueError("ChoiceForge bundle file changed: " + relative)
    metadata = json.loads((bundle / "adapter.json").read_text())
    if metadata.get("adapter_sha256") != manifest["files"]["adapter.safetensors"]:
        raise ValueError("ChoiceForge adapter metadata checksum differs")
    config = json.loads((bundle / "parent/model.json").read_text())
    verify_base_for_config(config, base_path)
    parent = make_judge(
        config, base_path=base_path, checkpoint=bundle / "parent", device=device)
    architecture = metadata["architecture"]
    adapter = build_merged_residual_adapter(
        architecture["hidden_size"], architecture["total_width"],
        architecture["layer_norm_eps"])
    state = load_file(str(bundle / "adapter.safetensors"), device="cpu")
    adapter.load_state_dict(state)
    return MergedDecisionModel(parent, adapter, metadata)
