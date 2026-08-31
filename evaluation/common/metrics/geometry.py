"""Tier 7 -- representation geometry.  Tag: ratio (explanatory).

These do not rank models. They explain *why* a model wins Tier 1: a low-rank or
anisotropic latent is trivially easier to predict, so a good one-step NMSE on
such a space is worth less. Essential context for LeWM, whose encoder trains
from scratch and may still be collapsing.

All operate on encoded (real) latents only -- no predictor involved.
"""
from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-12


def _pool(latents: Tensor) -> Tensor:
    """``(B, T, P, D)`` -> ``(N, D)``: one vector per (clip, frame), mean over tokens.
    Already-``(N, D)`` input passes through."""
    if latents.dim() == 4:
        return latents.mean(dim=2).reshape(-1, latents.shape[-1]).float()
    return latents.reshape(-1, latents.shape[-1]).float()


def _covariance(x: Tensor) -> Tensor:
    x = (x - x.mean(dim=0, keepdim=True)).double()
    return (x.T @ x) / max(x.shape[0] - 1, 1)


def effective_rank(latents: Tensor) -> dict[str, float]:
    """Participation ratio of the covariance spectrum:
    ``(sum lambda)^2 / sum(lambda^2)``.

    Ranges from 1 (all variance in one direction -- collapsed) to ``D`` (white).
    ``effective_rank_ratio`` divides by ``D`` so it is comparable across models
    with different latent widths.
    """
    ev = torch.linalg.eigvalsh(_covariance(_pool(latents))).clamp_min(0.0)
    s1, s2 = ev.sum(), ev.pow(2).sum()
    er = (s1 * s1 / s2.clamp_min(_EPS)).item()
    d = latents.shape[-1]
    return {"effective_rank": er, "effective_rank_ratio": er / d, "latent_dim": d}


def variance_utilisation(latents: Tensor, floor_frac: float = 0.01) -> dict[str, float]:
    """Fraction of dims whose variance exceeds ``floor_frac`` x mean variance.
    Dead dims are wasted capacity."""
    var = _pool(latents).var(dim=0)
    alive = (var > floor_frac * var.mean()).float().mean().item()
    return {"variance_utilisation": alive, "dead_dim_fraction": 1.0 - alive}


def isotropy(latents: Tensor) -> dict[str, float]:
    """How tightly the latents are confined to a common direction / cone.

    ``isotropy = 1 - ||mean of unit-normalised latents||^2`` (no centring). If the
    tokens point every which way the unit vectors cancel and isotropy -> 1; if
    they sit in a narrow cone around a shared direction -- the anisotropy DINOv2
    is known for -- the mean unit vector has norm ~1 and isotropy -> 0. Bounded
    in [0, 1] and numerically stable, unlike the exp-partition form. Complements
    :func:`effective_rank`, which sees only the centred (shape) anisotropy.
    """
    x = _pool(latents).double()
    x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)
    common = (x.mean(dim=0) ** 2).sum().item()
    return {"isotropy": 1.0 - common}


def geometry_report(latents: Tensor) -> dict[str, float]:
    return {
        **effective_rank(latents),
        **variance_utilisation(latents),
        **isotropy(latents),
    }
