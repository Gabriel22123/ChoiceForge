"""Pinned causal decoder with a shared scalar candidate head; no generation."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .core import digest, validate_request, write_json
from .output_contract import validate_prediction
from .prompting import render_decoder_admission, render_decoder_candidate


MODEL_ID = "openbmb/MiniCPM5-2B-Base"
MODEL_REVISION = "96a57cd572a02506b4500f54427dca24970c1bac"


class DecoderJudge:
    """Score independently rendered candidates using final-token states."""

    def __init__(self, config, base_path=None, checkpoint=None, device="auto"):
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        self.torch = torch
        if config["model_id"] != MODEL_ID or config["revision"] != MODEL_REVISION:
            raise ValueError("Decoder path supports only the pinned MiniCPM5-2B-Base checkpoint")
        if config.get("architecture") != "causal_decoder":
            raise ValueError("Decoder config must declare causal_decoder architecture")
        if config.get("candidate_encoding") != "independent":
            raise ValueError("Causal decoder requires independent candidate encoding")
        if config.get("candidate_chunk_size", 8) < 1:
            raise ValueError("Candidate chunk size must be positive")
        if config.get("inference_encoding", "path") not in ("path", "shared", "auto"):
            raise ValueError("Inference encoding must be path, shared or auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.config = config
        self.work = {"encoder_calls": 0, "encoder_sequences": 0, "unpadded_tokens": 0,
                     "padded_token_slots": 0, "attention_token_pairs": 0,
                     "shared_prefix_requests": 0, "shared_prefix_tokens": 0}
        torch.manual_seed(config["seed"])
        dtype = torch.float32 if device == "cpu" else torch.float16
        source = str(base_path) if base_path else config["model_id"]
        kwargs = {"local_files_only": True} if base_path else {"revision": config["revision"]}
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
        base, info = AutoModel.from_pretrained(
            source, trust_remote_code=False, use_safetensors=True, dtype=dtype,
            attn_implementation="sdpa", output_loading_info=True, **kwargs)
        unexpected = [key for key in info.get("unexpected_keys", []) if not key.startswith("lm_head.")]
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs") or unexpected:
            raise ValueError("Decoder base failed native compatibility checks: " + str(info))
        if base.config.model_type != "llama":
            raise ValueError("Pinned decoder no longer resolves to native LlamaModel")
        base.config.use_cache = False
        base.requires_grad_(False)
        layers = config["lora_layers"]
        if not 1 <= layers <= base.config.num_hidden_layers:
            raise ValueError("Invalid layer count")
        layer_pattern = "|".join(str(index) for index in range(base.config.num_hidden_layers - layers,
                                                                 base.config.num_hidden_layers))
        targets = (rf"(?:model\.)?layers\.(?:{layer_pattern})\."
                   rf"(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
        lora = LoraConfig(r=config["lora_rank"], lora_alpha=2 * config["lora_rank"],
                          lora_dropout=config["dropout"], bias="none", target_modules=targets)
        self.encoder = get_peft_model(base, lora)
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = nn.Sequential(nn.LayerNorm(base.config.hidden_size),
                                  nn.Linear(base.config.hidden_size, config["head_width"]), nn.GELU(),
                                  nn.Dropout(config["dropout"]), nn.Linear(config["head_width"], 1)).float()
        # Record the adapter parameter set before an optional continual-learning
        # route freezes it.  Frozen decision parameters still belong to the
        # checkpoint and must be saved even though the optimizer cannot update
        # them.
        self._encoder_decision_names = tuple(
            name for name, parameter in self.encoder.named_parameters()
            if parameter.requires_grad
        )
        residual_width = config.get("residual_head_width", 0)
        if (not isinstance(residual_width, int) or isinstance(residual_width, bool) or
                residual_width < 0):
            raise ValueError("Residual head width must be a nonnegative integer")
        self.residual_head = None
        if residual_width:
            self.residual_head = nn.Sequential(
                nn.LayerNorm(base.config.hidden_size),
                nn.Linear(base.config.hidden_size, residual_width), nn.GELU(),
                nn.Dropout(config["dropout"]), nn.Linear(residual_width, 1),
            ).float()
            # The expert is an additive correction.  A zero final projection
            # makes the composed function exactly equal to the parent before
            # its first update, independent of the hidden representation.
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
        self.parent_decision_frozen = config.get("freeze_parent_decision_function", False)
        if not isinstance(self.parent_decision_frozen, bool):
            raise ValueError("Parent decision freeze control must be boolean")
        self.track_frozen_parent_gradients = config.get("track_frozen_parent_gradients", False)
        if not isinstance(self.track_frozen_parent_gradients, bool):
            raise ValueError("Frozen parent gradient tracking control must be boolean")
        if self.track_frozen_parent_gradients and not self.parent_decision_frozen:
            raise ValueError("Frozen parent gradient tracking requires a frozen parent")
        if self.parent_decision_frozen and self.residual_head is None:
            raise ValueError("Freezing the parent decision function requires a residual head")
        if self.parent_decision_frozen:
            self.head.requires_grad_(False)
            if not self.track_frozen_parent_gradients:
                self.encoder.requires_grad_(False)
                # Checkpointing trades extra forward work for activation memory
                # and is useful only when gradients traverse the encoder.
                self.encoder.gradient_checkpointing_disable()
        self.encoder.to(self.device)
        self.head.to(self.device)
        if self.residual_head is not None:
            self.residual_head.to(self.device)
        self.temperature = 1.0
        if checkpoint:
            from safetensors.torch import load_file
            path = Path(checkpoint)
            if (path / "checksums.json").exists():
                for filename, expected in json.loads((path / "checksums.json").read_text()).items():
                    if Path(filename).name != filename or digest((path / filename).read_bytes()) != expected:
                        raise ValueError("Checkpoint integrity check failed")
            state = load_file(str(path / "decision.safetensors"))
            params = dict(self.named_decision_parameters())
            checkpoint_config = json.loads((path / "model.json").read_text())
            checkpoint_has_residual = bool(checkpoint_config.get("residual_head_width", 0))
            expected = (set(params) if checkpoint_has_residual else
                        {name for name in params if not name.startswith("residual_head.")})
            if set(state) != expected:
                raise ValueError("Checkpoint parameter set mismatch")
            with torch.no_grad():
                for name, parameter in params.items():
                    if name not in state:
                        continue
                    parameter.copy_(state[name].to(device=self.device, dtype=parameter.dtype))
            if (path / "calibration.json").exists():
                self.temperature = json.loads((path / "calibration.json").read_text())["temperature"]
        self.train(False)

    def named_decision_parameters(self):
        encoder = dict(self.encoder.named_parameters())
        yield from (("encoder." + name, encoder[name]) for name in self._encoder_decision_names)
        yield from (("head." + name, parameter) for name, parameter in self.head.named_parameters())
        if self.residual_head is not None:
            yield from (("residual_head." + name, parameter)
                        for name, parameter in self.residual_head.named_parameters())

    def named_trainable(self):
        for name, parameter in self.named_decision_parameters():
            if not parameter.requires_grad:
                continue
            # A frozen parent may retain autograd solely to keep accelerator
            # execution on the stable training path.  It remains outside every
            # optimizer group; only the additive expert is trainable.
            if self.parent_decision_frozen and not name.startswith("residual_head."):
                continue
            yield name, parameter

    def clear_frozen_parent_gradients(self):
        """Discard autograd-only parent gradients before the next update."""
        if not getattr(self, "track_frozen_parent_gradients", False):
            return
        for parameter in self.encoder.parameters():
            parameter.grad = None

    def train(self, mode=True):
        track_parent = (self.parent_decision_frozen and
                        getattr(self, "track_frozen_parent_gradients", False))
        self.encoder.train(mode if track_parent else
                           (False if self.parent_decision_frozen else mode))
        if track_parent and mode:
            # Keep the encoder on its proven accelerator training graph while
            # making the frozen parent function deterministic.  Checkpointing
            # keys off module training state; Dropout is the only intended
            # stochastic component and is disabled recursively here.
            for module in self.encoder.modules():
                if isinstance(module, self.torch.nn.Dropout):
                    module.train(False)
        self.head.train(False if self.parent_decision_frozen else mode)
        if self.residual_head is not None:
            self.residual_head.train(mode)

    def encode(self, request):
        validate_request(request)
        admission = self.tokenizer.encode(render_decoder_admission(request), add_special_tokens=True)
        if len(admission) > self.config["max_tokens"]:
            raise ValueError(f"Input too long: {len(admission)} > {self.config['max_tokens']}; no truncation")
        sequences = []
        for index in range(len(request["choices"])):
            ids = self.tokenizer.encode(render_decoder_candidate(request, index), add_special_tokens=True)
            if len(ids) > self.config["max_tokens"]:
                raise ValueError("Input too long in decoder candidate encoding; no truncation")
            sequences.append((ids, [len(ids) - 1]))
        return sequences

    def logits(self, sequences):
        inference_encoding = self.config.get("inference_encoding", "path")
        if not self.encoder.training and inference_encoding != "path":
            rows = []
            for request in sequences:
                prefix = self._common_prefix(request)
                use_shared = inference_encoding == "shared" or (
                    len(request) >= self.config.get("shared_prefix_min_candidates", 8)
                    and prefix >= self.config.get("shared_prefix_min_tokens", 128)
                )
                rows.append(self._shared_prefix_logits(request, prefix) if use_shared and prefix > 0
                            else self._candidate_logits(request))
            count = max(map(len, rows))
            return self.torch.stack([
                self.torch.nn.functional.pad(row, (0, count - len(row)), value=-1e4)
                for row in rows
            ])
        flat = [candidate for request in sequences for candidate in request]
        chunk = self.config.get("candidate_chunk_size", 8) if self.encoder.training else 1
        values = self.torch.cat([self._candidate_logits(flat[index:index + chunk])
                                 for index in range(0, len(flat), chunk)])
        scores, offset = [], 0
        count = max(map(len, sequences))
        for request in sequences:
            row = values[offset:offset + len(request)]
            scores.append(self.torch.nn.functional.pad(row, (0, count - len(request)), value=-1e4))
            offset += len(request)
        return self.torch.stack(scores)

    def features(self, sequences):
        """Return padded final-token candidate features without applying the head.

        This path is intentionally limited to independent private paths.  It is
        used by representation-frozen objective studies, where the expensive
        public base is evaluated once and every compared objective receives the
        exact same candidate vectors.  Zero rows are padding, never candidates.
        """
        flat = [candidate for request in sequences for candidate in request]
        chunk = self.config.get("candidate_chunk_size", 8)
        values = self.torch.cat([self._candidate_features(flat[index:index + chunk])
                                 for index in range(0, len(flat), chunk)])
        rows, offset = [], 0
        count = max(map(len, sequences))
        for request in sequences:
            row = values[offset:offset + len(request)]
            rows.append(self.torch.nn.functional.pad(
                row, (0, 0, 0, count - len(request)), value=0.0))
            offset += len(request)
        return self.torch.stack(rows)

    @staticmethod
    def _common_prefix(sequences):
        """Return a token prefix shared by every path, leaving a suffix token."""
        if not sequences:
            return 0
        paths = [sequence for sequence, _ in sequences]
        limit = min(map(len, paths)) - 1
        for index in range(max(0, limit)):
            if any(path[index] != paths[0][index] for path in paths[1:]):
                return index
        return max(0, limit)

    @staticmethod
    def _repeat_cache(cache, count):
        """Copy one causal prefix cache into independent candidate branches."""
        cache = copy.deepcopy(cache)
        if hasattr(cache, "batch_repeat_interleave"):
            cache.batch_repeat_interleave(count)
            return cache
        return tuple(tuple(value.repeat_interleave(count, dim=0) for value in layer)
                     for layer in cache)

    def _shared_prefix_logits(self, sequences, prefix_length):
        """Score private candidate suffixes against one cached common prefix."""
        torch = self.torch
        if any(markers != [len(sequence) - 1] for sequence, markers in sequences):
            return self._candidate_logits(sequences)
        prefix = torch.tensor([sequences[0][0][:prefix_length]], dtype=torch.long,
                              device=self.device)
        root = self.encoder(input_ids=prefix, attention_mask=torch.ones_like(prefix), use_cache=True)
        root_cache = root.past_key_values
        del root
        self.work["encoder_calls"] += 1
        self.work["encoder_sequences"] += 1
        self.work["unpadded_tokens"] += prefix_length
        self.work["padded_token_slots"] += prefix_length
        self.work["attention_token_pairs"] += prefix_length * (prefix_length + 1) // 2
        self.work["shared_prefix_requests"] = self.work.get("shared_prefix_requests", 0) + 1
        self.work["shared_prefix_tokens"] = self.work.get("shared_prefix_tokens", 0) + prefix_length

        values = []
        chunk_size = self.config.get("candidate_chunk_size", 8)
        for start in range(0, len(sequences), chunk_size):
            chunk = sequences[start:start + chunk_size]
            suffixes = [sequence[prefix_length:] for sequence, _ in chunk]
            length = max(map(len, suffixes))
            tokens = torch.full((len(chunk), length), self.tokenizer.pad_token_id,
                                dtype=torch.long, device=self.device)
            attention = torch.ones((len(chunk), prefix_length + length), dtype=torch.long,
                                   device=self.device)
            positions = []
            for index, suffix in enumerate(suffixes):
                tokens[index, :len(suffix)] = torch.tensor(suffix, device=self.device)
                attention[index, prefix_length + len(suffix):] = 0
                positions.append(len(suffix) - 1)
            cache = self._repeat_cache(root_cache, len(chunk))
            hidden = self.encoder(input_ids=tokens, attention_mask=attention,
                                  past_key_values=cache, use_cache=False).last_hidden_state
            rows = torch.arange(len(chunk), device=self.device)
            values.append(self._score_features(
                hidden[rows, torch.tensor(positions, device=self.device)].float()))
            self.work["encoder_calls"] += 1
            self.work["encoder_sequences"] += len(chunk)
            self.work["unpadded_tokens"] += sum(map(len, suffixes))
            self.work["padded_token_slots"] += len(chunk) * length
            self.work["attention_token_pairs"] += len(chunk) * (
                prefix_length * length + length * (length + 1) // 2)
            del hidden, cache
        del root_cache
        return torch.cat(values)

    def _candidate_features(self, sequences):
        torch = self.torch
        length = max(len(sequence) for sequence, _ in sequences)
        self.work["encoder_calls"] += 1
        self.work["encoder_sequences"] += len(sequences)
        self.work["unpadded_tokens"] += sum(len(sequence) for sequence, _ in sequences)
        self.work["padded_token_slots"] += len(sequences) * length
        self.work["attention_token_pairs"] += len(sequences) * length * (length + 1) // 2
        tokens = torch.full((len(sequences), length), self.tokenizer.pad_token_id,
                            dtype=torch.long, device=self.device)
        mask = torch.zeros_like(tokens)
        positions = []
        for index, (sequence, markers) in enumerate(sequences):
            tokens[index, :len(sequence)] = torch.tensor(sequence, device=self.device)
            mask[index, :len(sequence)] = 1
            positions.append(markers[0])
        hidden = self.encoder(input_ids=tokens, attention_mask=mask).last_hidden_state
        row_indices = torch.arange(len(sequences), device=self.device)
        return hidden[row_indices, torch.tensor(positions, device=self.device)].float()

    def _candidate_logits(self, sequences):
        return self._score_features(self._candidate_features(sequences))

    def _score_features(self, features):
        scores = self.head(features).squeeze(-1)
        if self.residual_head is not None:
            scores = scores + self.residual_head(features).squeeze(-1)
        return scores

    def predict(self, request):
        validate_request(request)
        self.train(False)
        with self.torch.no_grad():
            probabilities = (self.logits([self.encode(request)]) / self.temperature).softmax(-1)[0].cpu().tolist()
        if not all(__import__("math").isfinite(value) for value in probabilities):
            raise ValueError("Non-finite predictions")
        ids = [choice["id"] for choice in request["choices"]]
        return validate_prediction({
            "choice_id": ids[max(range(len(probabilities)), key=probabilities.__getitem__)],
            "probabilities": dict(zip(ids, probabilities)), "requires_review": True,
            "score_kind": "candidate_probability_not_a_guarantee",
            "calibration": "validation_temperature" if self.temperature != 1 else "uncalibrated",
        }, request)

    def save(self, path):
        from safetensors.torch import save_file
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({name: parameter.detach().cpu().contiguous()
                   for name, parameter in self.named_decision_parameters()},
                  str(path / "decision.safetensors"))
        write_json(path / "model.json", self.config)


def verify_local_decoder_base(path):
    """Verify every pinned decoder file against its local download lock."""
    from hashlib import file_digest
    lock = json.loads(Path(__file__).with_name("decoder_base.lock.json").read_text())
    if lock["revision"] != MODEL_REVISION:
        raise ValueError("Decoder lock revision differs")
    local_lock = Path(path) / "base.lock.json"
    if local_lock.is_file() and json.loads(local_lock.read_text()) != lock:
        raise ValueError("Local decoder lock differs from packaged lock")
    for name, expected in lock["files"].items():
        with (Path(path) / name).open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Local decoder base differs from pinned public original: " + name)
