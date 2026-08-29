"""Tier 2 -- open-loop rollout & temporal stability.  Tag: ratio.

The model is fed its own predictions. Absolute rollout error depends on the
latent's scale, so everything reported here is either a self-normalised ratio
(drift) or a unitless integer / slope, all directly cross-model.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .predictive import nmse

_EPS = 1e-8


def _log_slope(ys: list[float]) -> float:
    """OLS slope of ``log(y)`` against ``h = 1, 2, ...`` -- the exponential drift
    constant. 0 => error is flat with horizon."""
    xs = list(range(1, len(ys) + 1))
    ly = [math.log(max(y, _EPS)) for y in ys]
    n = len(xs)
    sx, sy = sum(xs), sum(ly)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ly))
    denom = n * sxx - sx * sx
    return (n * sxy - sx * sy) / denom if denom else 0.0


def path_straightness(latents: Tensor) -> float:
    """Mean turning angle (radians) between consecutive step vectors along a
    latent trajectory ``(B, T, P, D)``.

    On the *real* path: how linearly predictable the representation is -- a
    straighter path is a friendlier target for a world model (LeWM, Maes 2026).
    On a *rollout*: whether the model preserves trajectory geometry or kinks it.
    ``0`` = perfectly straight; ``pi/2`` = each step orthogonal to the last.
    """
    v = (latents[:, 1:] - latents[:, :-1]).flatten(2)      # (B, T-1, P*D)
    a, b = v[:, :-1], v[:, 1:]
    cos = F.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
    return torch.arccos(cos).mean().item()


def stability_from_nmse(nmse_by_h: dict[int, float]) -> dict:
    """Derive the horizon-stability metrics from an already-computed
    ``{horizon: nmse}`` map (e.g. streamed over the full val set). Same outputs as
    :func:`rollout_report` minus the straightness terms, which need the tensors."""
    hs = sorted(nmse_by_h)
    n1 = nmse_by_h[hs[0]]
    r2_h = {h: 1.0 - nmse_by_h[h] for h in hs}
    return {
        "nmse_by_h": {h: nmse_by_h[h] for h in hs},
        "r2_by_h": r2_h,
        "drift_ratio_by_h": {h: nmse_by_h[h] / max(n1, _EPS) for h in hs},
        "usable_horizon": max((h for h in hs if r2_h[h] >= 0.5), default=0),
        "compounding_rate": _log_slope([nmse_by_h[h] for h in hs]),
    }


def rollout_report(pred: Tensor, real: Tensor) -> dict:
    """``pred``, ``real``: ``(B, H, P, D)`` aligned by horizon (``h = 1..H``).

    ``real`` is the encoder's true latent for each rolled-out frame.
    """
    H = pred.shape[1]
    nmse_h = {h + 1: nmse(pred[:, h], real[:, h]) for h in range(H)}
    r2_h = {h: 1.0 - v for h, v in nmse_h.items()}
    n1 = nmse_h[1]

    return {
        "nmse_by_h": nmse_h,
        "r2_by_h": r2_h,
        # self-normalised growth: flat -> stable, steep -> diverging. Cross-model.
        "drift_ratio_by_h": {h: v / max(n1, _EPS) for h, v in nmse_h.items()},
        # one integer per model: largest horizon still explaining half the variance.
        "usable_horizon": max((h for h, v in r2_h.items() if v >= 0.5), default=0),
        # exponential drift constant.
        "compounding_rate": _log_slope([nmse_h[h] for h in sorted(nmse_h)]),
        "straightness_pred": path_straightness(pred),
        "straightness_real": path_straightness(real),
    }
