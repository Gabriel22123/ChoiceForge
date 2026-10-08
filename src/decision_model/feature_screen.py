"""Representation-frozen candidate-head utilities for objective screening."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .core import digest


class CandidateHead:
    """Factory wrapper kept import-light until torch extras are requested."""

    @staticmethod
    def build(hidden_size, width, dropout=0.0):
        from torch import nn
        return nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, width),
                             nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))


class RoutedResidualHead:
    """Permutation-invariant request router plus candidate residual scorer."""

    @staticmethod
    def build(hidden_size, residual_width, router_width, dropout=0.0,
              initial_gate_probability=0.5):
        import math
        from torch import nn
        if (hidden_size < 1 or residual_width < 1 or router_width < 1 or
                not 0.0 < initial_gate_probability < 1.0):
            raise ValueError("Invalid routed residual dimensions or initial gate")

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.residual = CandidateHead.build(hidden_size, residual_width, dropout)
                self.router = nn.Sequential(
                    nn.LayerNorm(hidden_size), nn.Linear(hidden_size, router_width),
                    nn.GELU(), nn.Dropout(dropout), nn.Linear(router_width, 1))
                # Exact parent equivalence comes from a zero residual, while a
                # finite gate keeps both paths differentiable after the first
                # residual update.
                nn.init.zeros_(self.residual[-1].weight)
                nn.init.zeros_(self.residual[-1].bias)
                nn.init.zeros_(self.router[-1].weight)
                nn.init.constant_(
                    self.router[-1].bias,
                    math.log(initial_gate_probability / (1.0 - initial_gate_probability)))

            def forward(self, features, mask):
                if (features.ndim != 3 or mask.ndim != 2 or
                        features.shape[:2] != mask.shape or
                        not bool(mask.any(dim=1).all())):
                    raise ValueError("Routed residual expects nonempty padded candidate rows")
                weights = mask.to(dtype=features.dtype).unsqueeze(-1)
                request = (features * weights).sum(dim=1) / weights.sum(dim=1)
                gate = self.router(request).squeeze(-1).sigmoid()
                residual = self.residual(features).squeeze(-1).masked_fill(~mask, 0.0)
                return residual * gate.unsqueeze(-1), gate

        return Module()


class FeatureStore:
    def __init__(self, path, require_formal=True):
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if self.manifest.get("status") != "complete":
            raise ValueError("Feature cache is incomplete")
        frozen = self.manifest.get("frozen", {})
        if frozen.get("format") not in {
                "typed-decisions-minicpm5-candidate-features-v1",
                "mbpp-outcome-minicpm5-candidate-features-v1"}:
            raise ValueError("Unknown feature cache format")
        if require_formal and not frozen.get("formal"):
            raise ValueError("Formal screen refuses a capped feature cache")
        if len(self.manifest.get("rows", {})) != frozen.get("row_count"):
            raise ValueError("Feature cache row index is incomplete")
        indexed = [row_id for shard in self.manifest["shards"] for row_id in shard["row_ids"]]
        if len(indexed) != len(set(indexed)) or set(indexed) != set(self.manifest["rows"]):
            raise ValueError("Feature shard membership differs from row index")
        for shard in self.manifest["shards"]:
            if digest((self.path / shard["file"]).read_bytes()) != shard["sha256"]:
                raise ValueError("Feature shard checksum mismatch: " + shard["file"])
            for row_id in shard["row_ids"]:
                entry = self.manifest["rows"][row_id]
                if (entry.get("shard") != shard["file"] or len(entry.get("shape", [])) != 2 or
                        entry["shape"][1] != self.manifest.get("hidden_size") or
                        len(entry.get("candidate_ids", [])) != entry["shape"][0]):
                    raise ValueError("Feature row metadata is inconsistent: " + row_id)
        self.manifest_sha256 = digest((self.path / "manifest.json").read_bytes())
        self._features = {}

    def load_subset(self, row_ids):
        requested = list(row_ids)
        if len(requested) != len(set(requested)) or any(row_id not in self.manifest["rows"]
                                                       for row_id in requested):
            raise ValueError("Requested feature IDs are duplicated or unknown")
        missing = set(requested) - set(self._features)
        from safetensors import safe_open
        for shard in self.manifest["shards"]:
            selected = [row_id for row_id in shard["row_ids"] if row_id in missing]
            if not selected:
                continue
            expected_keys = {self.manifest["rows"][row_id]["key"] for row_id in shard["row_ids"]}
            with safe_open(str(self.path / shard["file"]), framework="pt", device="cpu") as values:
                if set(values.keys()) != expected_keys:
                    raise ValueError("Feature tensor keys differ from manifest: " + shard["file"])
                for row_id in selected:
                    entry = self.manifest["rows"][row_id]
                    value = values.get_tensor(entry["key"])
                    if list(value.shape) != entry["shape"]:
                        raise ValueError("Feature shape mismatch: " + row_id)
                    self._features[row_id] = value
        if any(row_id not in self._features for row_id in requested):
            raise ValueError("Requested feature IDs were not loaded")
        return {row_id: self._features[row_id] for row_id in requested}

    def load_all(self):
        return self.load_subset(self.manifest["rows"])


def parameter_sha256(module):
    value = hashlib.sha256()
    for name, parameter in module.named_parameters():
        value.update(name.encode()); value.update(parameter.detach().float().cpu().numpy().tobytes())
    return value.hexdigest()


def padded_logits(head, feature_rows, permutations=None, device="cpu"):
    import torch
    if not feature_rows:
        raise ValueError("Expected nonempty feature batch")
    permutations = ([list(range(len(row))) for row in feature_rows]
                    if permutations is None else permutations)
    if len(permutations) != len(feature_rows):
        raise ValueError("Permutation batch differs from features")
    width = max(len(row) for row in feature_rows)
    hidden = feature_rows[0].shape[1]
    values = torch.zeros((len(feature_rows), width, hidden), dtype=torch.float32, device=device)
    mask = torch.zeros((len(feature_rows), width), dtype=torch.bool, device=device)
    for index, (row, order) in enumerate(zip(feature_rows, permutations)):
        if sorted(order) != list(range(len(row))):
            raise ValueError("Invalid candidate permutation")
        selected = row[order].to(device=device, dtype=torch.float32)
        values[index, :len(row)] = selected; mask[index, :len(row)] = True
    logits = head(values).squeeze(-1).masked_fill(~mask, -1e4)
    return logits, mask


def aligned_targets(rows, permutations, width, mode, device="cpu"):
    import torch
    from .core import target_distribution
    if mode not in ("hard", "distribution") or len(rows) != len(permutations):
        raise ValueError("Invalid target mode or permutation batch")
    if mode == "hard":
        targets = []
        for row, order in zip(rows, permutations):
            ids = [row["request"]["choices"][index]["id"] for index in order]
            targets.append(ids.index(row["label"]))
        return torch.tensor(targets, dtype=torch.long, device=device)
    targets = torch.zeros((len(rows), width), dtype=torch.float32, device=device)
    for index, (row, order) in enumerate(zip(rows, permutations)):
        choices = [row["request"]["choices"][candidate] for candidate in order]
        values = target_distribution(row, choices)
        targets[index, :len(values)] = torch.tensor(values, device=device)
    return targets
