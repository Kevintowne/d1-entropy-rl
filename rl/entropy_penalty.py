# rl/entropy_penalty.py
"""
Sample-based policy entropy penalty for PPO/GRPO.

Usage inside your trainer loss:
    from rl.entropy_penalty import add_entropy_penalty
    loss = add_entropy_penalty(loss, logprobs_new_tokens, action_mask, global_step, cfg)
"""

from types import SimpleNamespace
import math
import torch

def _get_cfg(cfg):
    # accept dict, SimpleNamespace, or any object with attributes
    if isinstance(cfg, dict):
        return SimpleNamespace(**cfg)
    return cfg

def sample_entropy_estimate(logprobs_new_tokens: torch.Tensor,
                            action_mask: torch.Tensor) -> torch.Tensor:
    """
    Unbiased entropy estimator using sampled tokens.
    logprobs_new_tokens: [B, T] log π(a_t | s_t) for chosen tokens
    action_mask:         [B, T] 1 on generated tokens, 0 on prompt/padding
    """
    neg_logp = -logprobs_new_tokens
    return (neg_logp * action_mask).sum() / (action_mask.sum() + 1e-8)

def schedule_coef(step: int, max_coef: float, *,
                  total_steps: int | None = None,
                  kind: str = "linear",
                  warmup: int = 0) -> float:
    if max_coef <= 0:
        return 0.0
    if kind == "none":
        return max_coef
    if kind == "linear":
        if warmup <= 0:
            return max_coef
        return max_coef * min(1.0, step / float(warmup))
    if kind == "cosine" and total_steps and total_steps > 0:
        x = min(step, total_steps) / total_steps
        return max_coef * (1.0 - math.cos(math.pi * x)) / 2.0
    return max_coef

def add_entropy_penalty(loss: torch.Tensor,
                        logprobs_new_tokens: torch.Tensor,
                        action_mask: torch.Tensor,
                        step: int,
                        cfg) -> torch.Tensor:
    """
    Adds +λ(t) * Entropy to the loss (minimization => penalize high entropy).
    Required cfg fields (as attr or keys):
        - entropy_penalty_coef (float)
        - entropy_penalty_schedule: "linear" | "cosine" | "none" (default: linear)
        - entropy_warmup_steps (int, default 0)
        - total_updates / total_steps (optional, for cosine)
    """
    cfg = _get_cfg(cfg)
    coef_max = float(getattr(cfg, "entropy_penalty_coef", 0.0))
    if coef_max <= 0:
        return loss
    sched = getattr(cfg, "entropy_penalty_schedule", "linear")
    warmup = int(getattr(cfg, "entropy_warmup_steps", 0))
    total = getattr(cfg, "total_updates", getattr(cfg, "total_steps", None))
    coef = schedule_coef(step, coef_max, total_steps=total, kind=sched, warmup=warmup)

    ent = sample_entropy_estimate(logprobs_new_tokens, action_mask)
    return loss + coef * ent
