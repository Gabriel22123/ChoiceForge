"""Native EuroBERT + LoRA + a shared candidate scorer; no text generation."""
import json
from pathlib import Path

from .core import digest, render, validate_request, write_json
from .output_contract import validate_prediction

MODEL_ID = "EuroBERT/EuroBERT-2.1B"
MODEL_REVISION = "81245a4d71f43452badf5e04458e4ddb831ff109"
BASE_WEIGHT_SHA256 = "95a20d9e222bb5edfaaddec384ccfd999cf9c9a4e3add3101df0d29b5731791e"


class Judge:
    def __init__(self, config, base_path=None, checkpoint=None, device="auto"):
        import torch
        from torch import nn
        from transformers import AutoModel, AutoTokenizer
        from peft import LoraConfig, get_peft_model
        self.torch = torch
        if config["model_id"] != MODEL_ID or config["revision"] != MODEL_REVISION:
            raise ValueError("This release supports the pinned native EuroBERT-2.1B checkpoint")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        self.config = config
        if config.get("candidate_encoding", "joint") not in ("joint", "independent"):
            raise ValueError("Unknown candidate encoding")
        if config.get("candidate_chunk_size", 8) < 1:
            raise ValueError("Candidate chunk size must be positive")
        self.work = {"encoder_calls": 0, "encoder_sequences": 0,
                     "unpadded_tokens": 0, "padded_token_slots": 0,
                     "attention_token_pairs": 0}
        torch.manual_seed(config["seed"])
        dtype = torch.float32 if device == "cpu" else torch.float16
        source = str(base_path) if base_path else config["model_id"]
        kwargs = {"local_files_only": True} if base_path else {"revision": config["revision"]}
        self.tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False, **kwargs)
        base, info = AutoModel.from_pretrained(source, trust_remote_code=False, use_safetensors=True,
                                              dtype=dtype, attn_implementation="sdpa", output_loading_info=True, **kwargs)
        unexpected = [k for k in info.get("unexpected_keys", []) if not k.startswith("lm_head.")]
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs") or unexpected:
            raise ValueError("Base checkpoint failed native compatibility checks: " + str(info))
        if base.config.is_decoder:
            raise ValueError("Expected a bidirectional encoder")
        base.config.use_cache = False
        base.requires_grad_(False)
        layers = config["lora_layers"]
        if not 1 <= layers <= base.config.num_hidden_layers:
            raise ValueError("Invalid layer count")
        layer_pattern = "|".join(str(i) for i in range(base.config.num_hidden_layers-layers,base.config.num_hidden_layers))
        targets = rf"(?:model\.)?layers\.(?:{layer_pattern})\.(?:self_attn|mlp)\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
        cfg = LoraConfig(r=config["lora_rank"], lora_alpha=2*config["lora_rank"],
                         lora_dropout=config["dropout"], bias="none",
                         target_modules=targets)
        self.encoder = get_peft_model(base, cfg)
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.head = nn.Sequential(nn.LayerNorm(base.config.hidden_size),
                                  nn.Linear(base.config.hidden_size, config["head_width"]), nn.GELU(),
                                  nn.Dropout(config["dropout"]), nn.Linear(config["head_width"], 1)).float()
        self.encoder.to(self.device); self.head.to(self.device)
        self.temperature = 1.0
        if checkpoint:
            from safetensors.torch import load_file
            path = Path(checkpoint)
            if (path/"checksums.json").exists():
                for filename, expected in json.loads((path/"checksums.json").read_text()).items():
                    if Path(filename).name != filename or digest((path/filename).read_bytes()) != expected:
                        raise ValueError("Checkpoint integrity check failed")
            state = load_file(str(path/"decision.safetensors"))
            params = dict(self.named_trainable())
            if set(state) != set(params):
                raise ValueError("Checkpoint parameter set mismatch")
            with torch.no_grad():
                for name, p in params.items():
                    p.copy_(state[name].to(device=self.device, dtype=p.dtype))
            if (path/"calibration.json").exists():
                self.temperature = json.loads((path/"calibration.json").read_text())["temperature"]
        self.train(False)

    def named_trainable(self):
        yield from (("encoder."+n, p) for n, p in self.encoder.named_parameters() if p.requires_grad)
        yield from (("head."+n, p) for n, p in self.head.named_parameters())

    def train(self, mode=True):
        self.encoder.train(mode); self.head.train(mode)

    def encode(self, request):
        # Both modes apply the same full-request token-budget admission rule.
        # Independent scoring must not silently gain different train/test cases.
        ids = self.tokenizer.encode(render(request,self.tokenizer.mask_token), add_special_tokens=True)
        if len(ids) > self.config["max_tokens"]:
            raise ValueError(f'Input too long: {len(ids)} > {self.config["max_tokens"]}; no truncation')
        positions = [i for i,v in enumerate(ids) if v==self.tokenizer.mask_token_id]
        if len(positions) != len(request["choices"]):
            raise ValueError("Reserved token collision")
        if self.config.get("candidate_encoding", "joint") == "independent":
            pairs = []
            for index in range(len(request["choices"])):
                pair = self.tokenizer.encode(render(request, self.tokenizer.mask_token, index), add_special_tokens=True)
                markers = [i for i, token in enumerate(pair) if token == self.tokenizer.mask_token_id]
                if len(markers) != 1:
                    raise ValueError("Reserved token collision")
                if len(pair) > self.config["max_tokens"]:
                    raise ValueError("Input too long in independent encoding; no truncation")
                pairs.append((pair, markers))
            return pairs
        return ids, positions

    def logits(self, sequences):
        if self.config.get("candidate_encoding", "joint") == "independent":
            flat = [pair for request in sequences for pair in request]
            # A single candidate per inference call makes padding/batch composition
            # independent of the caller's candidate order. Training can batch them.
            chunk = self.config.get("candidate_chunk_size", 8) if self.encoder.training else 1
            values = self.torch.cat([self._joint_logits(flat[i:i+chunk])[:, 0]
                                    for i in range(0, len(flat), chunk)])
            scores, offset = [], 0
            count = max(map(len, sequences))
            for request in sequences:
                row = values[offset:offset+len(request)]
                scores.append(self.torch.nn.functional.pad(row, (0, count-len(request)), value=-1e4))
                offset += len(request)
            return self.torch.stack(scores)
        return self._joint_logits(sequences)

    def _joint_logits(self, sequences):
        torch = self.torch
        length = max(len(seq) for seq,_ in sequences)
        if hasattr(self, "work"):
            self.work["encoder_calls"] += 1
            self.work["encoder_sequences"] += len(sequences)
            self.work["unpadded_tokens"] += sum(len(seq) for seq, _ in sequences)
            self.work["padded_token_slots"] += len(sequences) * length
            self.work["attention_token_pairs"] += len(sequences) * length * length
        x = torch.full((len(sequences), length), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(x)
        for i, (seq, _) in enumerate(sequences):
            x[i,:len(seq)] = torch.tensor(seq, device=self.device)
            mask[i,:len(seq)] = 1
        h = self.encoder(input_ids=x, attention_mask=mask).last_hidden_state
        # Every candidate uses the same head. Its meaning comes from its description,
        # not a learned class index. Padding is excluded from the choice distribution.
        count = max(len(pos) for _,pos in sequences)
        scores = []
        for i,(_,positions) in enumerate(sequences):
            logits = self.head(h[i,positions].float()).squeeze(-1)
            scores.append(torch.nn.functional.pad(logits,(0,count-len(positions)),value=-1e4))
        return torch.stack(scores)

    def predict(self, request):
        validate_request(request)
        self.train(False)
        with self.torch.no_grad():
            p = (self.logits([self.encode(request)])/self.temperature).softmax(-1)[0].cpu().tolist()
        if not all(__import__("math").isfinite(v) for v in p):
            raise ValueError("Non-finite predictions")
        ids = [c["id"] for c in request["choices"]]
        return validate_prediction({"choice_id":ids[max(range(len(p)),key=p.__getitem__)],
                                    "probabilities":dict(zip(ids,p)),"requires_review":True,
                                    "score_kind":"candidate_probability_not_a_guarantee",
                                    "calibration":"validation_temperature" if self.temperature != 1 else "uncalibrated"},
                                   request)

    def save(self, path):
        from safetensors.torch import save_file
        path = Path(path); path.mkdir(parents=True, exist_ok=True)
        state = {n:p.detach().cpu().contiguous() for n,p in self.named_trainable()}
        save_file(state, str(path/"decision.safetensors"))
        write_json(path/"model.json", self.config)


def verify_local_base(path):
    """A local override is only allowed to contain the original public checkpoint."""
    from hashlib import file_digest
    lock=json.loads(Path(__file__).with_name("base.lock.json").read_text())
    for name,expected in lock["files"].items():
        with (Path(path)/name).open("rb") as f:
            if file_digest(f,"sha256").hexdigest()!=expected:
                raise ValueError("Local base differs from pinned public original: "+name)
