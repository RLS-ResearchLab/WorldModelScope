"""Tier 5 -- representation probing against BridgeData ``state``.  Tag: ratio.

Do predicted latents still carry task information, or do they only "look
plausible"? Probe the latent for the robot's end-effector state; then check
whether a probe fit on *real* latents still reads out from *predicted* ones.

The BridgeData ``state`` vector ships with the canonical clip stream, so this
tier costs one extra ridge solve -- no labels to collect.
"""
from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-12


def _pool(z: Tensor) -> Tensor:
    """(N, P, D) -> (N, D); (N, D) passes through."""
    return z.mean(dim=1).float() if z.dim() == 3 else z.float()


def fit_probe(z_train: Tensor, y_train: Tensor, ridge: float = 1.0) -> Tensor:
    x = _pool(z_train).double()
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=-1)
    y = y_train.double()
    f = x.shape[1]
    return torch.linalg.solve(x.T @ x + ridge * torch.eye(f, dtype=x.dtype, device=x.device), x.T @ y)


def probe_r2(w: Tensor, z_eval: Tensor, y_eval: Tensor) -> dict[str, float]:
    x = _pool(z_eval).double()
    x = torch.cat([x, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)], dim=-1)
    y = y_eval.double()
    resid = y - x @ w
    ss_res = resid.pow(2).sum(dim=0)
    ss_tot = (y - y.mean(dim=0)).pow(2).sum(dim=0).clamp_min(_EPS)
    per_dim = (1.0 - ss_res / ss_tot)
    return {"probe_r2_mean": per_dim.mean().item(),
            "probe_r2_per_dim": [v.item() for v in per_dim]}


def linear_probe(
    z_train: Tensor, y_train: Tensor, z_eval: Tensor, y_eval: Tensor, ridge: float = 1.0
) -> dict[str, float]:
    """Fit on ``z_train`` (real latents), evaluate on ``z_eval``.

    * ``z_eval == real held-out latents``     -> how much task state the
      representation exposes linearly.
    * ``z_eval == predicted latents``, same episodes -> how much of that survives
      one prediction step.
    """
    w = fit_probe(z_train, y_train, ridge)
    return probe_r2(w, z_eval, y_eval)


def probe_transfer(r2_predicted: float, r2_real: float) -> float:
    """``R2(predicted) / R2(real)``. 1 => predictions keep the task info intact;
    ``< 1`` => semantically degraded even where they look plausible."""
    return r2_predicted / max(r2_real, _EPS)
