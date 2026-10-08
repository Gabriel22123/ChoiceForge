"""Public-data training and held-out evaluation; no internal dataset dependencies."""
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from .core import (canonical, digest, distribution_metrics, load_rows, metrics,
                   summarize_rows, target_distribution, write_json)
from .device_memory import AllocationSampler
from .judge_factory import make_judge, verify_base_for_config


def balanced_subset(rows, cap):
    if not cap or len(rows) <= cap:
        return sorted(rows, key=lambda r: r["id"])
    buckets = defaultdict(list)
    for row in sorted(rows, key=lambda r:r["id"]):
        buckets[(row["source"], row["family"], row["label"])].append(row)
    selected = []
    source_keys = {s:[k for k in sorted(buckets) if k[0]==s] for s in sorted({r["source"] for r in rows})}
    offsets = defaultdict(int)
    while len(selected) < cap:
        active = False
        for source,keys in source_keys.items():
            remaining = [k for k in keys if buckets[k]]
            if remaining:
                key=remaining[offsets[source]%len(remaining)]; offsets[source]+=1
                selected.append(buckets[key].pop(0)); active=True
                if len(selected)==cap: break
        if not active:
            break
    return selected


def predict_rows(judge, rows, tokens):
    judge.train(False)
    logits = []
    with judge.torch.no_grad():
        # Serial reference evaluation avoids batch-size-dependent near-tie decisions.
        for row in rows:
            logits.append(judge.logits([tokens[row["id"]]])[0].float().cpu().tolist())
    return logits


def evaluation_splits(splits, config):
    """Optionally bound the training probe; always retain all held-out rows."""
    cap = config.get("max_train_evaluation_rows", 0)
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 0:
        raise ValueError("Training evaluation cap must be a nonnegative integer")
    return dict(splits, train=balanced_subset(splits["train"], cap)) if cap else dict(splits)


def validate_epoch_orders(rows, config):
    """Optional frozen ordering only; every selected training row appears once per epoch."""
    orders = config.get("epoch_row_orders")
    if orders is None:
        return None
    if not isinstance(orders, list) or len(orders) != config["epochs"]:
        raise ValueError("Frozen ordering must specify every epoch")
    expected = Counter(r["id"] for r in rows)
    if any(not isinstance(order, list) or any(not isinstance(key, str) for key in order)
           or Counter(order) != expected for order in orders):
        raise ValueError("Frozen epoch must contain each selected training row exactly once")
    by_id = {r["id"]: r for r in rows}
    return [[by_id[key] for key in order] for order in orders]


def evaluate_rows(rows, logits, temperature=1):
    import torch
    p = [(torch.tensor(v)/temperature).softmax(-1).tolist() for v in logits]
    y = [[c["id"] for c in r["request"]["choices"]].index(r["label"]) for r in rows]
    has_soft = any("target_probabilities" in row for row in rows)
    targets = [target_distribution(row) for row in rows]
    result = distribution_metrics(targets, p) if has_soft else metrics(y,p)
    def subset(indices):
        return (distribution_metrics([targets[index] for index in indices], [p[index] for index in indices])
                if any("target_probabilities" in rows[index] for index in indices)
                else metrics([y[index] for index in indices], [p[index] for index in indices]))
    strata = {
        "by_source": "source", "by_family": "family",
        "by_evaluation_regime": "evaluation_regime", "by_decision_type": "decision_type",
    }
    for output_key, row_key in strata.items():
        values = sorted({row[row_key] for row in rows if row_key in row})
        result[output_key] = {
            value: subset([index for index, row in enumerate(rows) if row.get(row_key) == value])
            for value in values
        }
    predictions = []
    for row, values in zip(rows, p):
        point = {"id":row["id"], "label":row["label"],
                 "probabilities":dict(zip([c["id"] for c in row["request"]["choices"]], values))}
        if "decision_type" in row:
            point["decision_type"] = row["decision_type"]
        if "target_probabilities" in row:
            point["target_probabilities"] = row["target_probabilities"]
        predictions.append(point)
    return result, predictions


def calibrate(logits, rows):
    import torch
    values = torch.nn.utils.rnn.pad_sequence([torch.tensor(v,dtype=torch.float64) for v in logits],batch_first=True,padding_value=-1e4)
    y = torch.tensor([[c["id"] for c in r["request"]["choices"]].index(r["label"]) for r in rows])
    # Fixed grid includes identity; calibration never makes validation NLL worse.
    candidates = sorted({1.0, *[math.exp(-2.3 + i*4.6/200) for i in range(201)]})
    if any("target_probabilities" in row for row in rows):
        targets = torch.zeros_like(values)
        for index, row in enumerate(rows):
            distribution = target_distribution(row)
            targets[index, :len(distribution)] = torch.tensor(distribution, dtype=values.dtype)
        loss_for = lambda temperature: -(targets * torch.log_softmax(values / temperature, -1)).sum(-1).mean().item()
        method = "fixed log-spaced grid on validation soft cross entropy only"
    else:
        loss_for = lambda temperature: torch.nn.functional.cross_entropy(values / temperature, y).item()
        method = "fixed log-spaced grid on validation only"
    losses = [loss_for(t) for t in candidates]
    i = min(range(len(losses)), key=losses.__getitem__)
    return {"temperature": candidates[i], "validation_nll": losses[i],
            "identity_validation_nll": loss_for(1.0),
            "fit_rows":len(rows), "method":method,
            "test_used":False}


def training_targets(rows, views, width, device, target_mode="auto"):
    """Align supervision after candidate shuffling without changing hard runs."""
    import torch
    if target_mode not in ("auto", "hard", "distribution"):
        raise ValueError("Unknown training target mode")
    use_distribution = (target_mode == "distribution" or
                        (target_mode == "auto" and any("target_probabilities" in row for row in rows)))
    if target_mode == "distribution" and any("target_probabilities" not in row for row in rows):
        raise ValueError("Distribution target mode requires probabilities on every row")
    if not use_distribution:
        return torch.tensor([[choice["id"] for choice in view["choices"]].index(row["label"])
                             for row, view in zip(rows, views)], device=device)
    targets = torch.zeros((len(rows), width), dtype=torch.float32, device=device)
    for index, (row, view) in enumerate(zip(rows, views)):
        values = target_distribution(row, view["choices"])
        targets[index, :len(values)] = torch.tensor(values, dtype=torch.float32, device=device)
    return targets


def training_distribution_field(rows, views, width, device, field,
                                allow_missing=False):
    """Align an auxiliary probability distribution after candidate shuffling."""
    import torch
    targets = torch.zeros((len(rows), width), dtype=torch.float32, device=device)
    active = torch.zeros(len(rows), dtype=torch.float32, device=device)
    for index, (row, view) in enumerate(zip(rows, views)):
        values = row.get(field)
        if values is None and allow_missing:
            continue
        expected = {choice["id"] for choice in row["request"]["choices"]}
        if (not isinstance(values, dict) or set(values) != expected or
                any(not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or value < 0 for value in values.values()) or
                not math.isclose(sum(values.values()), 1.0, abs_tol=1e-5)):
            raise ValueError("Invalid auxiliary probability distribution: " + field)
        aligned = [float(values[choice["id"]]) for choice in view["choices"]]
        targets[index, :len(aligned)] = torch.tensor(
            aligned, dtype=torch.float32, device=device)
        active[index] = 1
    return (targets, active) if allow_missing else targets


def train(args):
    import torch
    from .objective import consistency_loss
    from .prior_correction import prior_only_request
    from .training_objective import (TrainingObjective, context_advantage_loss,
                                     distillation_kl_loss)
    config = json.loads(Path(args.config).read_text())
    objective = TrainingObjective(config)
    target_mode = config.get("training_target_mode", "auto")
    if target_mode not in ("auto", "hard", "distribution"):
        raise ValueError("Unknown training target mode")
    context_weight = config.get("context_advantage_weight", 0.0)
    context_gap = config.get("context_advantage_gap", 0.2)
    distillation_weight = config.get("distillation_weight", 0.0)
    distillation_field = config.get("distillation_target_field")
    distillation_allow_missing = config.get("distillation_allow_missing_targets", False)
    null_distillation_weight = config.get("null_distillation_weight", 0.0)
    null_distillation_field = config.get("null_distillation_target_field")
    null_distillation_allow_missing = config.get(
        "null_distillation_allow_missing_targets", False)
    training_weight_field = config.get("training_weight_field")
    if (not isinstance(context_weight, (int, float)) or isinstance(context_weight, bool) or
            not math.isfinite(context_weight) or context_weight < 0):
        raise ValueError("Context-advantage weight must be finite and nonnegative")
    if (not isinstance(context_gap, (int, float)) or isinstance(context_gap, bool) or
            not math.isfinite(context_gap) or context_gap < 0):
        raise ValueError("Context-advantage gap must be finite and nonnegative")
    if (not isinstance(distillation_weight, (int, float)) or
            isinstance(distillation_weight, bool) or
            not math.isfinite(distillation_weight) or distillation_weight < 0):
        raise ValueError("Distillation weight must be finite and nonnegative")
    if distillation_weight and (
            not isinstance(distillation_field, str) or not distillation_field):
        raise ValueError("Positive distillation weight requires a target field")
    if not isinstance(distillation_allow_missing, bool):
        raise ValueError("Distillation missing-target control must be boolean")
    if (not isinstance(null_distillation_weight, (int, float)) or
            isinstance(null_distillation_weight, bool) or
            not math.isfinite(null_distillation_weight) or null_distillation_weight < 0):
        raise ValueError("Null-context distillation weight must be finite and nonnegative")
    if null_distillation_weight and (
            not isinstance(null_distillation_field, str) or not null_distillation_field):
        raise ValueError("Positive null-context distillation weight requires a target field")
    if not isinstance(null_distillation_allow_missing, bool):
        raise ValueError("Null-context distillation missing-target control must be boolean")
    if training_weight_field is not None and (
            not isinstance(training_weight_field, str) or not training_weight_field):
        raise ValueError("Training-weight field must be a nonempty string")
    training_views = config.get("training_views", 2 if config.get("consistency_weight",0) else 1)
    if training_views not in (1,2) or (config.get("consistency_weight",0) and training_views!=2):
        raise ValueError("Consistency needs two training views; supported view counts: 1 or 2")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new run directory")
    rows = load_rows(args.data)
    if any(r.get("evaluation_regime")=="held_out_task" and r["split"]!="test" for r in rows):
        raise ValueError("Held-out tasks may not enter training or calibration")
    heldout={r["family"] for r in rows if r.get("evaluation_regime")=="held_out_task"}
    if any(r["family"] in heldout and r["split"]!="test" for r in rows):
        raise ValueError("Held-out task family leaked into training/validation")
    if args.base_path:
        verify_base_for_config(config, args.base_path)
    if args.init_from:
        previous = json.loads((Path(args.init_from)/"model.json").read_text())
        architecture_keys = ("model_id","revision","lora_layers","lora_rank","head_width")
        if (config.get("architecture", "bidirectional_encoder") !=
                previous.get("architecture", "bidirectional_encoder") or
                any(config[k] != previous[k] for k in architecture_keys)):
            raise ValueError("Warm-start architecture mismatch")
        if config.get("candidate_encoding", "joint") != previous.get("candidate_encoding", "joint"):
            raise ValueError("Warm-start candidate encoding mismatch; train a fresh architecture comparison")
    judge = make_judge(config, base_path=args.base_path, checkpoint=args.init_from, device=args.device)
    memory = AllocationSampler(torch, judge.device)
    memory.sample("model_loaded")
    tokens, too_long = {}, []
    for row in rows:
        try:
            tokens[row["id"]] = judge.encode(row["request"])
        except ValueError as exc:
            if "Input too long" not in str(exc):
                raise
            too_long.append(row["id"])
    available = [r for r in rows if r["id"] in tokens]
    if config.get("require_all_rows") and too_long:
        raise ValueError("Frozen dataset contains over-budget rows")
    splits = {s:balanced_subset([r for r in available if r["split"]==s], config.get("max_"+s+"_rows",0))
              for s in ("train","validation","test")}
    if any(not x for x in splits.values()):
        raise ValueError("All three splits must be nonempty")
    if training_weight_field is not None:
        for row in splits["train"]:
            value = row.get(training_weight_field)
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError("Every training row needs a finite positive training weight")
    frozen_orders = validate_epoch_orders(splits["train"], config)
    eval_splits = evaluation_splits(splits, config)
    output.mkdir(parents=True)
    run = {"dataset_sha256":digest(Path(args.data).read_bytes()), "config_sha256":digest(Path(args.config).read_bytes()),
           "initialization":"public_pretrained_base_fresh_adapter_and_head" if not args.init_from else "warm_start_new_optimizer",
           "warm_start_weights_sha256":digest((Path(args.init_from)/"decision.safetensors").read_bytes()) if args.init_from else None,
           "seed":config["seed"], "device":str(judge.device), "selected":summarize_rows(sum(splits.values(),[])),
           "selected_ids":{s:[r["id"] for r in v] for s,v in splits.items()},
           "evaluation_ids":{s:[r["id"] for r in v] for s,v in eval_splits.items()},
           "over_token_budget_ids":too_long, "checkpoint_selection":"fixed final epoch; test never used for selection",
           "trainable_parameters":sum(p.numel() for _,p in judge.named_trainable()),
           "base_parameters":sum(p.numel() for p in judge.encoder.parameters()),
           "head":"shared_candidate_scalar_no_fixed_classes","held_out_task_families":sorted(heldout),
           "runtime":{}, "source_files":{}}
    import importlib.metadata
    run["runtime"] = {p:importlib.metadata.version(p) for p in ("torch","transformers","peft","safetensors","numpy","jsonschema")}
    (output/"source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        raw = path.read_bytes(); (output/"source"/path.name).write_bytes(raw)
        run["source_files"][path.name] = digest(raw)
    write_json(output/"run.json",run)
    write_json(output/"model.json",config)
    params = list(judge.named_trainable())
    import hashlib
    initial_hash=hashlib.sha256()
    for name,p in params:
        initial_hash.update(name.encode());initial_hash.update(p.detach().float().cpu().numpy().tobytes())
    run["initial_trainable_sha256"]=initial_hash.hexdigest()
    run["selected_ids_sha256"]=digest(run["selected_ids"])
    run["training_views"]=training_views
    run["training_target_mode"] = target_mode
    run["forward_sequences"]=0
    run["forward_batches"]=0
    run["counterfactual_forward_sequences"]=0
    run["counterfactual_forward_batches"]=0
    plan_hash=hashlib.sha256()
    plan_path=output/"training-plan.jsonl"
    parameter_groups = []
    encoder_parameters = [p for n,p in params if n.startswith("encoder.")]
    head_parameters = [p for n,p in params if not n.startswith("encoder.")]
    if encoder_parameters:
        parameter_groups.append({"params": encoder_parameters, "lr": config["encoder_lr"]})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": config["head_lr"]})
    if not parameter_groups:
        raise ValueError("Training configuration exposes no trainable decision parameters")
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=0.01, eps=1e-6)
    rng = random.Random(config["seed"])
    microbatch, accumulation = config["batch_size"], config["accumulation"]
    updates = 0; curve = []; start = time.monotonic()
    gradient_norms = []
    context_penalty_sum = context_penalty_rows = 0
    distillation_penalty_sum = distillation_penalty_rows = 0
    null_distillation_penalty_sum = null_distillation_penalty_rows = 0
    fallback_probe = next(((n, p) for n, p in reversed(params) if p.ndim > 1), params[-1])
    probe_name, probe = next(((n,p) for n,p in params if "lora_B" in n), fallback_probe)
    initial_probe = probe.detach().cpu().clone()
    print(canonical({"event":"start", "trainable_parameters":run["trainable_parameters"],
                     "selected_counts":{s:len(v) for s,v in splits.items()},
                     "training_views":training_views,"initial_trainable_sha256":run["initial_trainable_sha256"]}),flush=True)
    for epoch in range(config["epochs"]):
        judge.train(True)
        if frozen_orders is None:
            order = splits["train"].copy(); rng.shuffle(order)
        else:
            order = frozen_orders[epoch]
        batches = [order[i:i+microbatch] for i in range(0,len(order),microbatch)]
        total_loss = total_surrogate = seen = 0
        for start_idx in range(0,len(batches),accumulation):
            group = batches[start_idx:start_idx+accumulation]
            optimizer.zero_grad(set_to_none=True)
            size = sum(map(len,group))
            for batch in group:
                views=[]
                for row in batch:
                    choices=row["request"]["choices"].copy(); rng.shuffle(choices)
                    views.append(dict(row["request"],choices=choices))
                x = [judge.encode(v) for v in views]
                logits = judge.logits(x)
                target = training_targets(batch, views, logits.shape[1], judge.device, target_mode)
                run["forward_sequences"]+=len(batch);run["forward_batches"]+=1
                counts = [len(view["choices"]) for view in views]
                kinds = [row.get("decision_type") for row in batch]
                sample_weights = (None if training_weight_field is None else torch.tensor(
                    [row[training_weight_field] for row in batch],
                    dtype=torch.float32, device=judge.device))
                candidate_orders = []
                for row, view in zip(batch, views):
                    semantic_ids = [choice["id"] for choice in row["request"]["choices"]]
                    candidate_orders.append([
                        semantic_ids.index(choice["id"]) for choice in view["choices"]])
                loss, risk = objective(logits, target, counts, kinds,
                                       candidate_orders=candidate_orders,
                                       sample_weights=sample_weights)
                if distillation_weight:
                    teacher = training_distribution_field(
                        batch, views, logits.shape[1], judge.device, distillation_field,
                        allow_missing=distillation_allow_missing)
                    teacher_target, teacher_active = ((teacher, None) if not
                        distillation_allow_missing else teacher)
                    distillation_penalty = distillation_kl_loss(
                        logits, teacher_target, counts, sample_weights=sample_weights,
                        active_mask=teacher_active)
                    loss = loss + float(distillation_weight) * distillation_penalty
                    risk = risk + float(distillation_weight) * distillation_penalty.detach()
                    active_rows = (len(batch) if teacher_active is None else
                                   int(teacher_active.sum().item()))
                    distillation_penalty_sum += distillation_penalty.detach().item() * active_rows
                    distillation_penalty_rows += active_rows
                if context_weight or null_distillation_weight:
                    # Eval mode makes the null forward deterministic.  The branch
                    # stays gradient-enabled only when its function is explicitly
                    # preserved by null-context distillation; the context-margin
                    # objective always sees a detached reference.
                    judge.train(False)
                    if null_distillation_weight:
                        null_logits = judge.logits([
                            judge.encode(prior_only_request(view)) for view in views])
                    else:
                        with torch.no_grad():
                            null_logits = judge.logits([
                                judge.encode(prior_only_request(view)) for view in views])
                    judge.train(True)
                    run["counterfactual_forward_sequences"] += len(batch)
                    run["counterfactual_forward_batches"] += 1
                if context_weight:
                    context_penalty = context_advantage_loss(
                        logits, null_logits, target, counts, context_gap,
                        sample_weights=sample_weights)
                    loss = loss + float(context_weight) * context_penalty
                    risk = risk + float(context_weight) * context_penalty.detach()
                    context_penalty_sum += context_penalty.detach().item() * len(batch)
                    context_penalty_rows += len(batch)
                if null_distillation_weight:
                    null_teacher = training_distribution_field(
                        batch, views, null_logits.shape[1], judge.device,
                        null_distillation_field,
                        allow_missing=null_distillation_allow_missing)
                    null_teacher_target, null_teacher_active = (
                        (null_teacher, None) if not null_distillation_allow_missing
                        else null_teacher)
                    null_distillation_penalty = distillation_kl_loss(
                        null_logits, null_teacher_target, counts,
                        sample_weights=sample_weights, active_mask=null_teacher_active)
                    loss = loss + float(null_distillation_weight) * null_distillation_penalty
                    risk = risk + float(null_distillation_weight) * null_distillation_penalty.detach()
                    active_rows = (len(batch) if null_teacher_active is None else
                                   int(null_teacher_active.sum().item()))
                    null_distillation_penalty_sum += (
                        null_distillation_penalty.detach().item() * active_rows)
                    null_distillation_penalty_rows += active_rows
                plan={"epoch":epoch+1,"rows":[r["id"] for r in batch],
                      "first":[[c["id"] for c in v["choices"]] for v in views]}
                if training_views==2:
                    others=[]
                    for view in views:
                        choices=view["choices"].copy(); rng.shuffle(choices)
                        others.append(dict(view,choices=choices))
                    other_raw=judge.logits([judge.encode(v) for v in others])
                    run["forward_sequences"]+=len(batch);run["forward_batches"]+=1
                    plan["second"]=[[c["id"] for c in v["choices"]] for v in others]
                    alignment=[]
                    for view,other_view in zip(views,others):
                        other_ids=[c["id"] for c in other_view["choices"]]
                        indices=[other_ids.index(c["id"]) for c in view["choices"]]
                        indices += list(range(len(indices),logits.shape[1]))
                        alignment.append(indices)
                    other=other_raw.gather(1,torch.tensor(alignment,device=judge.device))
                    other_loss, other_risk = objective(
                        other, target, counts, kinds, candidate_orders=candidate_orders,
                        sample_weights=sample_weights)
                    if distillation_weight:
                        other_teacher = training_distribution_field(
                            batch, others, other.shape[1], judge.device, distillation_field,
                            allow_missing=distillation_allow_missing)
                        other_teacher_target, other_teacher_active = ((other_teacher, None) if not
                            distillation_allow_missing else other_teacher)
                        other_distillation = distillation_kl_loss(
                            other, other_teacher_target, counts,
                            sample_weights=sample_weights, active_mask=other_teacher_active)
                        other_loss = other_loss + float(distillation_weight) * other_distillation
                        other_risk = (other_risk +
                                      float(distillation_weight) * other_distillation.detach())
                    if null_distillation_weight:
                        judge.train(False)
                        other_null = judge.logits([
                            judge.encode(prior_only_request(view)) for view in others])
                        judge.train(True)
                        run["counterfactual_forward_sequences"] += len(batch)
                        run["counterfactual_forward_batches"] += 1
                        other_null_teacher = training_distribution_field(
                            batch, others, other_null.shape[1], judge.device,
                            null_distillation_field,
                            allow_missing=null_distillation_allow_missing)
                        other_null_target, other_null_active = (
                            (other_null_teacher, None) if not null_distillation_allow_missing
                            else other_null_teacher)
                        other_null_penalty = distillation_kl_loss(
                            other_null, other_null_target, counts,
                            sample_weights=sample_weights, active_mask=other_null_active)
                        other_loss = (other_loss +
                                      float(null_distillation_weight) * other_null_penalty)
                        other_risk = (other_risk + float(null_distillation_weight) *
                                      other_null_penalty.detach())
                    penalty = config.get("consistency_weight",0)*consistency_loss(logits,other)
                    loss = (loss+other_loss)/2 + penalty
                    risk = (risk+other_risk)/2 + penalty.detach()
                encoded_plan=canonical(plan)+"\n";plan_hash.update(encoded_plan.encode())
                with plan_path.open("a") as plan_file:plan_file.write(encoded_plan)
                if not torch.isfinite(loss) or not torch.isfinite(risk):
                    raise ValueError("Non-finite loss")
                (loss*len(batch)/size).backward()
                total_loss += risk.item()*len(batch)
                total_surrogate += loss.item()*len(batch)
                seen += len(batch)
            memory.sample("after_backward", updates + 1)
            norm = torch.nn.utils.clip_grad_norm_([p for _,p in params],1.0,error_if_nonfinite=True)
            optimizer.step(); updates += 1
            if hasattr(judge, "clear_frozen_parent_gradients"):
                judge.clear_frozen_parent_gradients()
            gradient_norms.append(norm.item())
            with (output/"optimizer-steps.jsonl").open("a") as steps:
                steps.write(canonical({"epoch":epoch+1,"update":updates,"pre_clip_grad_norm":norm.item(),
                                       "clipped":norm.item()>1.0,"running_objective_risk":total_loss/seen,
                                       "running_surrogate":total_surrogate/seen})+"\n")
            if updates == 1 or updates%8 == 0:
                print(canonical({"event":"train","epoch":epoch+1,"update":updates,"loss":total_loss/seen,
                                 "optimization_surrogate":total_surrogate/seen,
                                 "grad_norm":norm.item(),"seconds":time.monotonic()-start}),flush=True)
        point = {"epoch":epoch+1,"updates":updates,"mean_training_loss":total_loss/seen,
                 "mean_optimization_surrogate":total_surrogate/seen}
        curve.append(point); write_json(output/"curve.json",curve)
        judge.save(output)
    run["updates"] = updates
    run["training_seconds"] = time.monotonic()-start
    run["encoder_work"] = dict(judge.work)
    run["sampled_device_allocation"] = memory.record()
    run["encoder_work_note"] = "Logical forward inputs only; attention pairs are an input-shape proxy, not FLOPs; excludes checkpoint recomputation and evaluation"
    run["training_plan_sha256"]=plan_hash.hexdigest()
    run["objective"]=objective.record()
    run["training_weighting"] = {
        "field": training_weight_field,
        "mean": (sum(float(row[training_weight_field]) for row in splits["train"]) /
                 len(splits["train"]) if training_weight_field else 1.0),
        "minimum": (min(float(row[training_weight_field]) for row in splits["train"])
                    if training_weight_field else 1.0),
        "maximum": (max(float(row[training_weight_field]) for row in splits["train"])
                    if training_weight_field else 1.0),
    }
    run["context_advantage"]={
        "weight":float(context_weight), "gap":float(context_gap),
        "null_branch_gradient":"stopped", "null_forward_mode":"eval",
        "mean_unweighted_training_penalty":(
            context_penalty_sum/context_penalty_rows if context_penalty_rows else None),
        "rows":context_penalty_rows,
        "purpose":"gold log-probability evidence gain over a null-context adaptive baseline",
    }
    run["functional_distillation"] = {
        "weight": float(distillation_weight),
        "target_field": distillation_field,
        "allow_missing_targets": distillation_allow_missing,
        "mean_training_kl": (distillation_penalty_sum / distillation_penalty_rows
                             if distillation_penalty_rows else None),
        "rows": distillation_penalty_rows,
        "gold_supervision_replaced": False,
        "purpose": "preserve a frozen teacher distribution while learning from gold labels",
    }
    run["null_functional_distillation"] = {
        "weight": float(null_distillation_weight),
        "target_field": null_distillation_field,
        "allow_missing_targets": null_distillation_allow_missing,
        "mean_training_kl": (
            null_distillation_penalty_sum / null_distillation_penalty_rows
            if null_distillation_penalty_rows else None),
        "rows": null_distillation_penalty_rows,
        "teacher_request": "same task and candidates with context replaced by an empty object",
        "null_forward_mode": "eval with gradients only when weight is positive",
        "purpose": "preserve the context-ablated teacher function used as an evidence baseline",
    }
    run["gradient_clipping"]={"max_norm":1.0,"updates":len(gradient_norms),
                              "clipped_updates":sum(value>1 for value in gradient_norms),
                              "mean_pre_clip_norm":sum(gradient_norms)/len(gradient_norms),
                              "max_pre_clip_norm":max(gradient_norms)}
    run["adapter_probe"] = {"parameter":probe_name,"absolute_change":(probe.detach().cpu()-initial_probe).abs().sum().item()}
    torch.save({"optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),"python_rng":rng.getstate(),
                "objective_noise_rng":objective.generator.get_state()},output/"training_state.pt")
    valid_logits = predict_rows(judge,splits["validation"],tokens)
    calibration = calibrate(valid_logits,splits["validation"])
    write_json(output/"calibration.json",calibration)
    evaluation = {}
    for split in ("train","validation","test"):
        eval_rows = eval_splits[split]
        logits = valid_logits if split == "validation" else predict_rows(judge,eval_rows,tokens)
        raw,_ = evaluate_rows(eval_rows,logits)
        calibrated,predictions = evaluate_rows(eval_rows,logits,calibration["temperature"])
        evaluation[split] = {"raw":raw,"temperature_scaled":calibrated}
        write_json(output/(split+"-predictions.json"),predictions)
    write_json(output/"evaluation.json",evaluation)
    write_json(output/"run.json",run)
    write_json(output/"checksums.json",{n:digest((output/n).read_bytes()) for n in
                                         ("model.json","decision.safetensors","calibration.json")})
    print(canonical({"event":"complete","updates":updates,"test":evaluation["test"]}),flush=True)
    return evaluation
