"""Tier 3 -- action grounding.  Tag: ratio.  Action-conditioned models only.

Does the predictor *use* the action, or is the conditioning cosmetic? The
encoder output is computed once; only the (cheap) predictor is re-run with
modified actions.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .predictive import nmse

_EPS = 1e-8


def shuffle_actions(actions: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Break the action<->transition pairing while keeping the marginal action
    distribution intact: permute whole action sequences along the batch axis."""
    b = actions.shape[0]
    perm = torch.randperm(b, generator=generator, device=actions.device)
    return actions[perm]


def ablation_report(nmse_real: float, nmse_shuffled: float, nmse_zero: float) -> dict[str, float]:
    """From three teacher-forced NMSEs (same clips, same encoder pass, different
    action inputs).

    ``action_reliance``      -- ``(NMSE_shuffled - NMSE_real) / NMSE_shuffled``.
                                0 => ignores actions; -> 1 => fully depends on them.
    ``action_effect_vs_zero``-- same against zeroed actions (isolates "no action"
                                from "wrong action").
    """
    return {
        "nmse_real_actions": nmse_real,
        "nmse_shuffled_actions": nmse_shuffled,
        "nmse_zero_actions": nmse_zero,
        "action_reliance": (nmse_shuffled - nmse_real) / max(nmse_shuffled, _EPS),
        "action_effect_vs_zero": (nmse_zero - nmse_real) / max(nmse_zero, _EPS),
    }


def counterfactual_divergence(
    pred_a: Tensor, pred_b: Tensor, real_a: Tensor, real_b: Tensor
) -> float:
    """Two action sequences from one shared start state:
    ``mean( ||pred_a - pred_b|| / ||real_a - real_b|| )``.

    ``~1`` calibrated response magnitude, ``<1`` under-responsive, ``>1``
    over-reactive. ``real_b`` is typically unavailable from a single logged
    trajectory -- the runner substitutes the encoder's own one-step delta as the
    reference scale and documents the substitution.
    """
    dp = (pred_a - pred_b).flatten(1).norm(dim=-1)
    dr = (real_a - real_b).flatten(1).norm(dim=-1)
    return (dp / dr.clamp_min(_EPS)).mean().item()


def action_inversion_r2(
    z_t: Tensor, z_next: Tensor, a_t: Tensor, train_frac: float = 0.8, ridge: float = 1.0
) -> float:
    """Fit a linear map ``(pool(z_t), pool(z_{t+1})) -> a_t`` by ridge regression
    on a train split, report mean per-dim R2 on the held-out split.

    High => the transition (the pair of latents) linearly encodes the action
    that produced it -- strong evidence the dynamics are action-grounded rather
    than the action being averaged away.

    ``z_t``, ``z_next``: ``(N, P, D)`` or ``(N, D)``. ``a_t``: ``(N, A)``.
    """
    def pool(z: Tensor) -> Tensor:
        return z.mean(dim=1).float() if z.dim() == 3 else z.float()

    x = torch.cat([pool(z_t), pool(z_next)], dim=-1).double()
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=-1)
    y = a_t.reshape(a_t.shape[0], -1).double()

    n_tr = int(train_frac * x.shape[0])
    xtr, xte = x[:n_tr], x[n_tr:]
    ytr, yte = y[:n_tr], y[n_tr:]
    if xte.shape[0] == 0:
        xte, yte = xtr, ytr

    f = xtr.shape[1]
    w = torch.linalg.solve(xtr.T @ xtr + ridge * torch.eye(f, dtype=x.dtype, device=x.device), xtr.T @ ytr)
    resid = yte - xte @ w
    ss_res = resid.pow(2).sum(dim=0)
    ss_tot = (yte - yte.mean(dim=0)).pow(2).sum(dim=0).clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).mean().item()
