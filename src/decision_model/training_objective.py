"""Model-training objectives with isolated, repexternal-projectble Gaussian draws."""
import hashlib
import math

import torch

from .objective import noisy_objective, proper_cost, proper_loss


def ranked_probability_cost(logits, targets):
    """Ranked probability score for one ordered candidate distribution."""
    if logits.shape[-1] < 2 or targets.shape[-1] != logits.shape[-1]:
        raise ValueError("RPS needs matching ordered distributions with >=2 levels")
    probabilities = torch.softmax(logits.float(), -1)
    return ((probabilities.cumsum(-1)[..., :-1] - targets.cumsum(-1)[..., :-1]) ** 2).sum(-1) / (
        logits.shape[-1] - 1)


def clean_cross_entropy(logits, targets):
    if targets.is_floating_point():
        return -(targets * torch.log_softmax(logits.float(), -1)).sum(-1).mean()
    return torch.nn.functional.cross_entropy(logits.float(), targets)


def _validated_sample_weights(logits, sample_weights):
    if sample_weights is None:
        return None
    if (sample_weights.ndim != 1 or len(sample_weights) != len(logits) or
            not torch.isfinite(sample_weights).all() or (sample_weights <= 0).any()):
        raise ValueError("Sample weights must be a finite positive vector matching the batch")
    return sample_weights.to(device=logits.device, dtype=logits.float().dtype)


def context_advantage_loss(logits, null_logits, targets, counts, gap=0.2,
                           sample_weights=None):
    """Require evidence to improve the gold log probability over null context.

    The null branch is detached deliberately.  It is an adaptive difficulty
    reference, not a target whose useful candidate semantics should be erased.
    """
    if logits.shape != null_logits.shape or len(counts) != len(logits):
        raise ValueError("Context-advantage logits or counts differ")
    if not isinstance(gap, (int, float)) or isinstance(gap, bool) or not math.isfinite(gap) or gap < 0:
        raise ValueError("Context-advantage gap must be finite and nonnegative")
    losses = []
    for index, count in enumerate(counts):
        if not 2 <= count <= logits.shape[1]:
            raise ValueError("Invalid context-advantage candidate count or target")
        evidence_logp = torch.log_softmax(logits[index, :count], dim=-1)
        baseline_logp = torch.log_softmax(null_logits[index, :count].detach(), dim=-1)
        if targets.is_floating_point():
            target = targets[index, :count]
            if (target < 0).any() or not torch.isfinite(target).all() or not torch.isclose(
                    target.sum(), target.new_tensor(1.0), atol=1e-5, rtol=0):
                raise ValueError("Invalid context-advantage probability target")
            evidence = (target * evidence_logp).sum()
            baseline = (target * baseline_logp).sum()
        else:
            target = int(targets[index])
            if not 0 <= target < count:
                raise ValueError("Invalid context-advantage candidate count or target")
            evidence = evidence_logp[target]
            baseline = baseline_logp[target]
        losses.append(torch.relu(logits.new_tensor(float(gap)) - (evidence - baseline)))
    values = torch.stack(losses)
    weights = _validated_sample_weights(logits, sample_weights)
    return values.mean() if weights is None else (values * weights).mean()


def distillation_kl_loss(logits, teacher_targets, counts, sample_weights=None,
                         active_mask=None):
    """Preserve a frozen teacher distribution without replacing gold labels."""
    if teacher_targets.shape != logits.shape or len(counts) != len(logits):
        raise ValueError("Distillation targets must match padded logits")
    if active_mask is None:
        active_mask = logits.new_ones(len(logits), dtype=torch.float32)
        masked = False
    else:
        masked = True
        if (active_mask.ndim != 1 or len(active_mask) != len(logits) or
                not torch.isfinite(active_mask).all() or
                ((active_mask != 0) & (active_mask != 1)).any()):
            raise ValueError("Distillation active mask must be a binary batch vector")
        active_mask = active_mask.to(device=logits.device, dtype=logits.float().dtype)
    losses = []
    for index, count in enumerate(counts):
        if not 2 <= count <= logits.shape[1]:
            raise ValueError("Invalid distillation candidate count")
        if not active_mask[index]:
            losses.append(logits[index, :count].sum() * 0)
            continue
        target = teacher_targets[index]
        if (not torch.isfinite(target).all() or (target < 0).any() or
                not torch.isclose(target[:count].sum(), target.new_tensor(1.0),
                                  atol=1e-5, rtol=0) or
                torch.count_nonzero(target[count:])):
            raise ValueError("Invalid distillation probability target or padded mass")
        log_target = torch.where(target[:count] > 0, target[:count].log(),
                                 target[:count].new_zeros(()))
        log_model = torch.log_softmax(logits[index, :count].float(), -1)
        losses.append((target[:count] * (log_target - log_model)).sum())
    values = torch.stack(losses)
    weights = _validated_sample_weights(logits, sample_weights)
    if not masked:
        return values.mean() if weights is None else (values * weights).mean()
    if not active_mask.any():
        return values.sum() * 0
    effective = active_mask if weights is None else active_mask * weights
    return (values * effective).sum() / active_mask.sum()


class TrainingObjective:
    """Keep sampled-risk reporting separate from the score-function surrogate."""

    def __init__(self, config):
        self.mode = config.get("gradient_estimator", "clean")
        self.beta = config.get("brier_weight", 0.5)
        self.sigma = config.get("noise_sigma", 0.4)
        self.samples = config.get("noise_samples", 16)
        self.seed = config.get("noise_seed", 913)
        self.project = config.get("center_score_logits", True)
        self.rps_weight = config.get("ranked_probability_weight", 0.0)
        self.clean_anchor_weight = config.get("clean_anchor_weight", 0.0)
        if self.mode not in ("clean", "pathwise", "score_function"):
            raise ValueError("Unknown gradient estimator")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("Brier weight must be finite and nonnegative")
        if (not isinstance(self.rps_weight, (int, float)) or isinstance(self.rps_weight, bool) or
                not math.isfinite(self.rps_weight) or self.rps_weight < 0):
            raise ValueError("Ranked-probability weight must be finite and nonnegative")
        if (not isinstance(self.clean_anchor_weight, (int, float)) or
                isinstance(self.clean_anchor_weight, bool) or
                not math.isfinite(self.clean_anchor_weight) or self.clean_anchor_weight < 0):
            raise ValueError("Clean-anchor weight must be finite and nonnegative")
        if self.mode == "clean" and self.clean_anchor_weight:
            raise ValueError("Clean-anchor weight is only defined for noisy estimators")
        if not math.isfinite(self.sigma) or self.sigma <= 0 or not isinstance(self.samples, int) or self.samples < 2:
            raise ValueError("Expected finite positive noise sigma and >=2 samples")
        self.generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.noise_hash = hashlib.sha256()
        self.calls = 0
        self.noise_elements = 0

    @staticmethod
    def _semantic_order(values, order, count):
        """Undo a presentation permutation before an order-sensitive cost.

        ``order[presented_index]`` is the original semantic candidate index.
        Proper CE/Brier losses are permutation invariant, whereas RPS is not.
        """
        if order is None:
            return values[..., :count]
        if len(order) != count or sorted(order) != list(range(count)):
            raise ValueError("Invalid semantic candidate order")
        inverse = [order.index(index) for index in range(count)]
        return values[..., inverse]

    def _rps(self, logits, targets, counts, kinds, candidate_orders):
        values = []
        for index, (count, kind, order) in enumerate(zip(counts, kinds, candidate_orders)):
            if kind != "score":
                continue
            if targets.is_floating_point():
                target = self._semantic_order(targets[index:index + 1], order, count)
            else:
                presented = torch.nn.functional.one_hot(targets[index:index + 1], count).float()
                target = self._semantic_order(presented, order, count)
            semantic_logits = self._semantic_order(logits[index:index + 1], order, count)
            values.append(ranked_probability_cost(semantic_logits, target).mean())
        return torch.stack(values).mean() if values else logits.new_zeros(())

    def __call__(self, logits, targets, counts, kinds=None, candidate_orders=None,
                 sample_weights=None):
        if len(counts) != len(logits) or any(not 2 <= k <= logits.shape[1] for k in counts):
            raise ValueError("Invalid candidate counts")
        kinds = [None] * len(counts) if kinds is None else kinds
        if len(kinds) != len(counts) or any(kind not in (None, "boolean", "choice", "score") for kind in kinds):
            raise ValueError("Invalid decision kinds")
        candidate_orders = ([None] * len(counts) if candidate_orders is None else candidate_orders)
        if len(candidate_orders) != len(counts):
            raise ValueError("Candidate-order batch differs")
        for count, order in zip(counts, candidate_orders):
            if order is not None and (len(order) != count or sorted(order) != list(range(count))):
                raise ValueError("Invalid semantic candidate order")
        if self.rps_weight and any(kind is None for kind in kinds):
            raise ValueError("Ranked-probability training requires decision_type metadata")
        if targets.is_floating_point():
            if targets.shape != logits.shape:
                raise ValueError("Probability targets must match padded logits")
            for target, count in zip(targets, counts):
                if (not torch.isfinite(target).all() or (target < 0).any() or
                        not torch.isclose(target[:count].sum(), target.new_tensor(1.0), atol=1e-5, rtol=0) or
                        torch.count_nonzero(target[count:])):
                    raise ValueError("Invalid probability target or padded mass")
        elif any(not 0 <= int(target) < k for target, k in zip(targets, counts)):
            raise ValueError("Target falls outside real candidates")
        weights = _validated_sample_weights(logits, sample_weights)
        if self.mode == "clean":
            # Preserve the legacy expression/gradient for existing configurations.
            loss = (proper_loss(logits, targets, self.beta) if weights is None else
                    (proper_cost(logits, targets, self.beta) * weights).mean())
            if self.rps_weight:
                if weights is not None:
                    raise ValueError("Sample weighting with ranked-probability loss is not supported")
                loss = loss + float(self.rps_weight) * self._rps(
                    logits, targets, counts, kinds, candidate_orders)
            return loss, loss.detach()
        if weights is not None:
            raise ValueError("Sample weighting is currently defined only for the clean objective")
        noise = torch.randn(self.samples, len(logits), logits.shape[1], generator=self.generator)
        self.noise_hash.update(str(tuple(noise.shape)).encode())
        self.noise_hash.update(noise.numpy().tobytes())
        self.calls += 1
        self.noise_elements += noise.numel()
        noise = noise.to(device=logits.device, dtype=logits.dtype)
        losses, risks = [], []
        for i, count in enumerate(counts):
            # Exclude padded classes from both reward and policy density.
            values = logits[i:i+1, :count]
            if self.project:
                # Removes the unidentifiable common-logit gradient direction.
                # Softmax risk is invariant to this translation.
                values = values - values.mean(-1, keepdim=True)
            eps = noise[:, i:i+1, :count]
            order = candidate_orders[i]
            if order is not None:
                # The dedicated draw plan is defined in semantic candidate order.
                # Present the same draw with the same permutation as the logits so
                # finite-sample noisy objectives remain exactly permutation stable.
                eps = eps[..., order]
            target = targets[i:i+1, :count] if targets.is_floating_point() else targets[i:i+1]
            losses.append(noisy_objective(values, target, eps, self.sigma, self.beta, self.mode))
            if self.rps_weight and kinds[i] == "score":
                actions = values.detach().unsqueeze(0) + self.sigma * eps
                rps_target = (target if target.is_floating_point() else
                              torch.nn.functional.one_hot(target, count).float())
                semantic_actions = self._semantic_order(actions, order, count)
                semantic_target = self._semantic_order(rps_target, order, count)
                rps_cost = ranked_probability_cost(semantic_actions, semantic_target)
                if self.mode == "pathwise":
                    losses[-1] = losses[-1] + float(self.rps_weight) * ranked_probability_cost(
                        self._semantic_order(values.unsqueeze(0) + self.sigma * eps, order, count),
                        semantic_target).mean()
                else:
                    baseline = (rps_cost.sum(0, keepdim=True) - rps_cost) / (len(eps) - 1)
                    log_density = -0.5 * ((actions - values.unsqueeze(0)) / self.sigma).square().sum(-1)
                    losses[-1] = losses[-1] + float(self.rps_weight) * (
                        (rps_cost - baseline) * log_density).mean()
            with torch.no_grad():
                risk = proper_cost(values.detach().unsqueeze(0) + self.sigma*eps, target, self.beta).mean()
                if self.rps_weight and kinds[i] == "score":
                    risk = risk + float(self.rps_weight) * ranked_probability_cost(
                        self._semantic_order(values.detach().unsqueeze(0) + self.sigma * eps,
                                             candidate_orders[i], count),
                        self._semantic_order(
                            target if target.is_floating_point() else
                            torch.nn.functional.one_hot(target, count).float(),
                            candidate_orders[i], count)).mean()
                risks.append(risk)
        loss, risk = torch.stack(losses).mean(), torch.stack(risks).mean()
        if self.clean_anchor_weight:
            anchor = clean_cross_entropy(logits, targets)
            loss = loss + float(self.clean_anchor_weight) * anchor
            risk = risk + float(self.clean_anchor_weight) * anchor.detach()
        return loss, risk

    def record(self):
        return {"gradient_estimator": self.mode, "noise_seed": self.seed,
                "noise_sigma": self.sigma, "noise_samples": self.samples,
                "center_score_logits": self.project,
                "ranked_probability_weight": float(self.rps_weight),
                "ordered_cost_candidate_order": (
                    "original semantic order restored after presentation permutation"
                    if self.rps_weight else None),
                "clean_anchor_weight": float(self.clean_anchor_weight),
                "noise_calls": self.calls, "noise_elements": self.noise_elements,
                "noise_plan_sha256": self.noise_hash.hexdigest() if self.calls else None,
                "noise_generator": "dedicated CPU torch.Generator; does not consume dropout RNG",
                "risk": ("clean CE+Brier+optional ordered RPS" if self.mode == "clean" else
                         "Monte Carlo Gaussian-smoothed CE+Brier+optional ordered RPS plus optional clean CE anchor"),
                "surrogate_is_risk": self.mode != "score_function",
                "score_baseline": "independent leave-one-out; no reward standard-deviation normalization"}
