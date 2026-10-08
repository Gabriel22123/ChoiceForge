"""Matched head-only fitting of frozen marker/mean features, with no-context controls."""
import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

from decision_model.core import digest, load_rows, metrics, write_json
from decision_model.model import Judge, verify_local_base
from decision_model.training_objective import TrainingObjective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", default="runs/overfit-sanity-v1")
    parser.add_argument("--output", default="runs/frozen-readout-v1")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(__file__).resolve().parents[1]
    source, output = root / args.sanity, root / args.output
    if output.exists():
        raise ValueError("Use a new directory")
    protocol = json.loads((source / "protocol.json").read_text())
    rows = load_rows(source / "cases.jsonl")
    if any(r["split"] != "train" for r in rows) or [r["id"] for r in rows] != protocol["selected_ids"]:
        raise ValueError("Probe requires exactly the sanity check's training rows")
    labels = sorted({r["label"] for r in rows})
    if len(rows) != 24 or sorted(Counter(r["label"] for r in rows).values()) != [8,8,8]:
        raise ValueError("Expected the balanced 24-row probe")
    initial = root / protocol["starting_checkpoint"]
    if digest((initial / "decision.safetensors").read_bytes()) != protocol["starting_weights_sha256"]:
        raise ValueError("Starting checkpoint changed")
    base = root / "models/eurobert-2.1b"
    verify_local_base(base)
    config = protocol["config"]
    judge = Judge(config, base_path=base, checkpoint=initial, device=args.device)
    initial_head = copy.deepcopy(judge.head).cpu()
    output.mkdir(parents=True)
    frozen = {"purpose": "Training-set head fitting on frozen features; not semantic/generalization evidence",
              "sanity_protocol_sha256": digest((source / "protocol.json").read_bytes()),
              "selected_ids": protocol["selected_ids"], "starting_weights_sha256": protocol["starting_weights_sha256"],
              "script_sha256": digest(Path(__file__).read_bytes()), "updates": 512, "batch": "all 24 rows",
              "learning_rate": config["head_lr"], "objective": "CE + .5 Brier", "dropout": 0.,
              "representations": ["candidate marker", "mean over all nonpadding tokens, including special tokens"],
              "negative_control": "Identical task/candidate meanings and constant context for every example",
              "limits": ["Only head parameters updated; encoder frozen", "Same initialization/optimizer/budget for all four arms",
                         "This has a different budget from the LoRA sanity experiment; no causal cross-experiment ranking",
                         "Memorizing 24 examples would not establish semantic understanding"]}
    write_json(output / "protocol.json", frozen)
    write_json(output / "status.json", {"state": "extracting"})
    feature_lists = {"marker": [], "mean": []}

    def extract(request):
        by_id = {c["id"]: c for c in request["choices"]}
        ordered = dict(request, choices=[by_id[label] for label in labels])
        values = {"marker": [], "mean": []}
        with torch.no_grad():
            for ids, positions in judge.encode(ordered):
                x = torch.tensor([ids], device=judge.device)
                hidden = judge.encoder(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state[0].float()
                values["marker"].append(hidden[positions[0]].cpu())
                values["mean"].append(hidden.mean(0).cpu())
        return {name: torch.stack(v) for name,v in values.items()}

    for row in rows:
        values = extract(row["request"])
        for name, value in values.items():
            feature_lists[name].append(value)
    removed = extract(dict(rows[0]["request"], context="No information is provided."))
    features = {name: torch.stack(v) for name,v in feature_lists.items()}
    features.update({name+"-no-context": value.unsqueeze(0).repeat(24,1,1) for name,value in removed.items()})
    targets = torch.tensor([labels.index(r["label"]) for r in rows])
    torch.save(dict(features, targets=targets), output / "features.pt")
    del judge
    gc.collect()
    if args.device == "mps":
        torch.mps.empty_cache()
    write_json(output / "status.json", {"state": "fitting_heads"})
    objective = TrainingObjective({"brier_weight": .5})
    result = {"protocol": frozen, "arms": {}}
    for name, x in features.items():
        head = copy.deepcopy(initial_head).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=config["head_lr"], weight_decay=.01, eps=1e-6)
        curve = []
        start = time.monotonic()
        for step in range(513):
            if step in (0, 1, 32, 128, 512):
                with torch.no_grad():
                    probabilities = head(x).squeeze(-1).softmax(-1)
                    metric = metrics(targets.tolist(), probabilities.tolist())
                    predicted = Counter(labels[i] for i in probabilities.argmax(-1).tolist())
                point = {"step": step, "metrics": metric, "predicted_labels": dict(predicted), "seconds": time.monotonic()-start}
                curve.append(point)
                print(json.dumps({"arm": name, "step": step, "correct": metric["correct"], "nll": metric["nll"]}), flush=True)
            if step == 512:
                break
            optimizer.zero_grad(set_to_none=True)
            logits = head(x).squeeze(-1)
            loss,_ = objective(logits, targets, [3]*24)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        if name.endswith("no-context") and curve[-1]["metrics"]["correct"] != 8:
            raise ValueError("Identical-input negative control exceeded/below constant-class accuracy")
        result["arms"][name] = {"curve": curve, "final": curve[-1], "initial": curve[0]}
        save_file(head.state_dict(), str(output / (name+"-head.safetensors")))
        write_json(output / "result.json", result)
    write_json(output / "status.json", {"state": "complete"})


if __name__ == "__main__":
    main()
