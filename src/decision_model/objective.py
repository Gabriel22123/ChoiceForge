"""Proper scoring with exact gradients; this is not a reproduction of proprietary RLCD."""
import torch


def proper_loss(logits, targets, beta=0.5):
    """CE + beta * expected multiclass Brier; accepts distributions or hard labels."""
    return proper_cost(logits,targets,beta).mean()


def proper_cost(logits, targets, beta=0.5):
    """Unreduced per-example proper loss, including broadcasted sample axes."""
    if not 0 <= beta:
        raise ValueError("beta must be nonnegative")
    if not targets.is_floating_point():
        targets = torch.nn.functional.one_hot(targets, logits.shape[-1]).float()
    values=logits if logits.dtype==torch.float64 else logits.float()
    logp = torch.log_softmax(values, -1)
    p = logp.exp()
    return -(targets * logp).sum(-1) + beta * (
        p.square().sum(-1) - 2 * (p * targets).sum(-1) + targets.square().sum(-1))


def consistency_loss(first, second):
    """JS between two stochastic views; output labels have fixed semantic positions."""
    p, q = first.float().softmax(-1), second.float().softmax(-1)
    m = ((p+q)/2).clamp_min(1e-8)
    return 0.5*((p*(p.clamp_min(1e-8).log()-m.log())).sum(-1) +
                (q*(q.clamp_min(1e-8).log()-m.log())).sum(-1)).mean()


def noisy_objective(logits, targets, noise, sigma=0.4, beta=0.5, estimator="pathwise"):
    """Two unbiased gradient estimators for the SAME Gaussian-smoothed risk.

    noise: [samples, batch, candidates], drawn independently from N(0, I).
    Score-function uses an independent leave-one-out baseline, not group reward
    standardization. The returned score-function value is a surrogate, not risk.
    This helper does not implement any private algorithm.
    """
    if sigma<=0 or noise.ndim!=3 or noise.shape[1:]!=logits.shape or len(noise)<2:
        raise ValueError("Expected sigma>0 and >=2 Gaussian samples matching logits")
    if estimator=="pathwise":
        return proper_cost(logits.unsqueeze(0)+sigma*noise,targets,beta).mean()
    if estimator!="score_function":
        raise ValueError("Unknown gradient estimator")
    actions=logits.detach().unsqueeze(0)+sigma*noise
    costs=proper_cost(actions,targets,beta).detach()
    baseline=(costs.sum(0,keepdim=True)-costs)/(len(noise)-1)
    log_density=-0.5*((actions-logits.unsqueeze(0))/sigma).square().sum(-1)
    return ((costs-baseline)*log_density).mean()
