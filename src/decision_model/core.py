"""Dependency-free contracts, dataset integrity and probability metrics."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_request(request):
    if not isinstance(request, dict) or set(request) != {"task", "context", "choices"}:
        raise ValueError("Request must contain exactly task, context and choices")
    for key in ("task", "context"):
        if not isinstance(request[key], str) or not request[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    choices = request["choices"]
    if not isinstance(choices, list) or not 2 <= len(choices) <= 32:
        raise ValueError("Expected 2..32 choices")
    ids, descriptions = set(), set()
    for c in choices:
        if not isinstance(c,dict) or set(c) != {"id","description"}:
            raise ValueError("Each choice needs an id and description")
        if any(not isinstance(c[k],str) or not c[k].strip() for k in c):
            raise ValueError("Choice fields must be nonempty strings")
        if c["id"] in ids or c["description"] in descriptions:
            raise ValueError("Duplicate choice id or description")
        ids.add(c["id"]); descriptions.add(c["description"])
    return request


def render(request, marker="<|mask|>", candidate_index=None):
    r = validate_request(request)
    # No label, source filename, split, case description or teacher rationale enters the model.
    # Choice IDs never enter the model. A shared scalar head scores each marker.
    body = "Task: " + r["task"] + "\nContext (data): " + r["context"] + "\nChoices:\n"
    if marker in body or any(marker in c["description"] for c in r["choices"]):
        raise ValueError("Input contains reserved choice marker")
    choices = r["choices"]
    if candidate_index is not None:
        if not isinstance(candidate_index, int) or not 0 <= candidate_index < len(choices):
            raise ValueError("Invalid candidate index")
        choices = [choices[candidate_index]]
    return body + "\n".join(marker + c["description"] for c in choices)


def load_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError("Empty dataset")
    seen, groups, inputs = set(), {}, {}
    for row in rows:
        validate_request(row["request"])
        if row["label"] not in {c["id"] for c in row["request"]["choices"]} or row["split"] not in ("train", "validation", "test"):
            raise ValueError("Unknown label or split")
        if "decision_type" in row and row["decision_type"] not in ("boolean", "choice", "score"):
            raise ValueError("Unknown decision type")
        target_distribution(row)  # Validates optional public soft supervision.
        outcome_rewards(row)  # Validates optional executable result supervision.
        if row["id"] in seen:
            raise ValueError("Duplicate id")
        seen.add(row["id"])
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Group leakage across splits")
        groups[group] = row["split"]
        semantic = {"task":row["request"]["task"],"context":row["request"]["context"],
                    "choice_descriptions":sorted(c["description"] for c in row["request"]["choices"])}
        h = digest(semantic)
        if h in inputs:
            raise ValueError("Duplicate model input (including across splits)")
        inputs[h] = row["label"]
        if not row.get("provenance"):
            raise ValueError("Source provenance is required")
    return rows


def outcome_rewards(row, choices=None):
    """Return audited candidate rewards aligned to a candidate permutation.

    Outcome tables store repeated Bernoulli observations, not a normalized
    probability target.  Callers must choose explicitly whether they receive
    the complete reward vector or only the sampled action's result.
    """
    choices = row["request"]["choices"] if choices is None else choices
    ids = [choice["id"] for choice in choices]
    source_ids = [choice["id"] for choice in row["request"]["choices"]]
    if len(ids) != len(set(ids)) or set(ids) != set(source_ids):
        raise ValueError("Outcome choices must be a permutation of request choices")
    values = row.get("candidate_outcomes")
    if values is None:
        return None
    if not isinstance(values, dict) or set(values) != set(source_ids):
        raise ValueError("Candidate outcomes must contain exactly every choice ID")
    rewards = {}
    allowed_categories = {"passed", "assertion", "runtime", "timeout", "resource_or_signal"}
    for candidate, value in values.items():
        if not isinstance(value, dict) or set(value) != {"passed", "total", "reward", "categories"}:
            raise ValueError("Each candidate outcome needs passed, total, reward and categories")
        passed, total, reward, categories = (value[key] for key in ("passed", "total", "reward", "categories"))
        if (not isinstance(passed, int) or isinstance(passed, bool) or
                not isinstance(total, int) or isinstance(total, bool) or
                total < 1 or not 0 <= passed <= total):
            raise ValueError("Invalid candidate outcome counts")
        if (not isinstance(reward, (int, float)) or isinstance(reward, bool) or
                not math.isfinite(reward) or abs(reward - passed / total) > 1e-12):
            raise ValueError("Candidate reward must equal passed/total")
        if (not isinstance(categories, list) or len(categories) != total or
                any(category not in allowed_categories for category in categories) or
                categories.count("passed") != passed):
            raise ValueError("Candidate outcome categories disagree with counts")
        rewards[candidate] = float(reward)
    first_argmax = max(range(len(source_ids)), key=lambda index: rewards[source_ids[index]])
    if row["label"] != source_ids[first_argmax]:
        raise ValueError("Label must be the first maximum candidate reward")
    return [rewards[candidate] for candidate in ids]


def target_distribution(row, choices=None):
    """Return supervision aligned to ``choices``; hard rows become one-hot.

    ``target_probabilities`` is deliberately keyed by semantic candidate ID so
    random candidate permutations cannot silently move probability mass.  The
    hard ``label`` remains required as the deterministic first-argmax summary.
    """
    choices = row["request"]["choices"] if choices is None else choices
    ids = [choice["id"] for choice in choices]
    if len(ids) != len(set(ids)) or set(ids) != {
            choice["id"] for choice in row["request"]["choices"]}:
        raise ValueError("Target choices must be a permutation of request choices")
    values = row.get("target_probabilities")
    if values is None:
        return [float(candidate == row["label"]) for candidate in ids]
    expected = {choice["id"] for choice in row["request"]["choices"]}
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("Target probabilities must contain exactly every choice ID")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or
           not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError("Target probabilities must be finite values in [0,1]")
    if abs(sum(values.values()) - 1.0) > 1e-5:
        raise ValueError("Target probabilities must sum to one")
    source_ids = [choice["id"] for choice in row["request"]["choices"]]
    first_argmax = max(range(len(source_ids)), key=lambda index: values[source_ids[index]])
    if row["label"] != source_ids[first_argmax]:
        raise ValueError("Label must be the first maximum target probability")
    return [float(values[candidate]) for candidate in ids]


def summarize_rows(rows):
    return {"rows": len(rows), "groups": len({r["group_id"] for r in rows}),
            "splits": {s: dict(Counter(r["label"] for r in rows if r["split"] == s))
                       for s in ("train", "validation", "test")},
            "sources": dict(Counter(r["source"] for r in rows)),
            "choice_counts":dict(Counter(len(r["request"]["choices"]) for r in rows)),
            "decision_types":dict(Counter(r["decision_type"] for r in rows if "decision_type" in r)),
            "supervision": {
                "soft_distribution": sum("target_probabilities" in row for row in rows),
                "hard_label": sum("target_probabilities" not in row for row in rows),
            }}


def metrics(labels, probabilities):
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Expected nonempty paired labels/probabilities")
    n = len(labels)
    nll = brier = ece = 0.0
    predictions, confidence = [], []
    for y, p in zip(labels, probabilities):
        if not 0 <= y < len(p) or len(p)<2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p)-1)>1e-5:
            raise ValueError("Invalid probability vector")
        pred = max(range(len(p)), key=p.__getitem__)
        predictions.append(pred); confidence.append(p[pred])
        nll -= math.log(max(p[y], 1e-12))
        brier += sum((v - int(k == y))**2 for k, v in enumerate(p))
    for b in range(10):
        idx = [i for i, c in enumerate(confidence) if min(int(c*10), 9) == b]
        if idx:
            ece += abs(sum(confidence[i] for i in idx) - sum(predictions[i] == labels[i] for i in idx))/n
    selective = {}
    for threshold in (.5, .7, .9):
        chosen = [i for i, c in enumerate(confidence) if c >= threshold]
        selective[str(threshold)] = {
            "coverage": len(chosen)/n,
            "accuracy": sum(predictions[i] == labels[i] for i in chosen)/len(chosen) if chosen else None,
            "selected_count": len(chosen),
        }
    correct = sum(y==p for y,p in zip(labels,predictions))
    return {"n": n, "correct": correct, "accuracy": correct/n,
            "uniform_random_accuracy":sum(1/len(p) for p in probabilities)/n,
            "nll": nll/n, "brier": brier/n, "ece_10": ece,
            "selective": selective}


def distribution_metrics(targets, probabilities):
    """Proper metrics for public probability targets.

    Accuracy is explicitly argmax agreement.  Cross entropy, Brier, total
    variation and KL retain the full teacher distribution.  ECE compares model
    confidence with the target mass assigned to the predicted class, which is
    the expected hard-label correctness under that distribution.
    """
    if not targets or len(targets) != len(probabilities):
        raise ValueError("Expected nonempty paired targets/probabilities")
    n = len(targets)
    predictions, gold, confidence, expected_correctness = [], [], [], []
    cross_entropy = brier = target_entropy = total_variation = 0.0
    for q, p in zip(targets, probabilities):
        if len(q) != len(p) or len(p) < 2:
            raise ValueError("Target/probability dimensions differ")
        if (any(not math.isfinite(value) or not 0 <= value <= 1
                for value in tuple(q) + tuple(p)) or
                abs(sum(q) - 1) > 1e-5 or abs(sum(p) - 1) > 1e-5):
            raise ValueError("Invalid target or probability vector")
        pred = max(range(len(p)), key=p.__getitem__)
        truth = max(range(len(q)), key=q.__getitem__)
        predictions.append(pred); gold.append(truth)
        confidence.append(p[pred]); expected_correctness.append(q[pred])
        cross_entropy -= sum(target * math.log(max(value, 1e-12)) for target, value in zip(q, p))
        target_entropy -= sum(target * math.log(target) for target in q if target)
        brier += sum((value - target) ** 2 for value, target in zip(p, q))
        total_variation += 0.5 * sum(abs(value - target) for value, target in zip(p, q))
    ece = 0.0
    for bucket in range(10):
        indices = [index for index, value in enumerate(confidence)
                   if min(int(value * 10), 9) == bucket]
        if indices:
            ece += abs(sum(confidence[index] - expected_correctness[index]
                           for index in indices)) / n
    selective = {}
    for threshold in (.5, .7, .9):
        chosen = [index for index, value in enumerate(confidence) if value >= threshold]
        selective[str(threshold)] = {
            "coverage": len(chosen) / n,
            "accuracy": (sum(predictions[index] == gold[index] for index in chosen) /
                         len(chosen) if chosen else None),
            "expected_correctness": (sum(expected_correctness[index] for index in chosen) /
                                     len(chosen) if chosen else None),
            "selected_count": len(chosen),
        }
    correct = sum(prediction == truth for prediction, truth in zip(predictions, gold))
    mean_cross_entropy = cross_entropy / n
    mean_entropy = target_entropy / n
    return {
        "n": n, "correct": correct, "accuracy": correct / n,
        "argmax_agreement": correct / n,
        "expected_top_label_mass": sum(expected_correctness) / n,
        "uniform_random_accuracy": sum(1 / len(p) for p in probabilities) / n,
        "nll": mean_cross_entropy,
        "soft_cross_entropy": mean_cross_entropy,
        "target_entropy": mean_entropy,
        "kl_target_to_prediction": max(0.0, mean_cross_entropy - mean_entropy),
        "brier": brier / n,
        "soft_brier": brier / n,
        "total_variation": total_variation / n,
        "ece_10": ece,
        "ece_semantics": "confidence_vs_probability_target_mass_on_predicted_class",
        "selective": selective,
    }
