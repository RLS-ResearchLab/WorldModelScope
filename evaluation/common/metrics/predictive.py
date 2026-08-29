"""Tier 1 -- scale-fair one-step prediction.  Tag: ratio.

Teacher-forced: real context in, predict the next latent, compare to the
encoder's real output for that frame. Every metric is dimensionless -- a ratio
that cancels the latent's per-dimension scale -- so a number from one model's
latent space is directly comparable to another's.

Divergence *between* these metrics is diagnostic: high R2 with poor cosine
means the scale is right but the direction is wrong.
"""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

_EPS = 1e-8


def _flat(x: Tensor) -> Tensor:
    """(..., D) -> (N, D)."""
    return x.reshape(-1, x.shape[-1]).float()


def nmse(pred: Tensor, target: Tensor) -> float:
    """Normalised MSE: ``||pred - target||^2 / ||target - mean(target)||^2``.

    * ``= 1.0``  -- as good as always predicting the target's (per-dim) mean.
    * ``< 1.0``  -- beats the mean predictor; equals the fraction of target
      variance left unexplained.
    * ``> 1.0``  -- worse than predicting the mean.

    The mean is per-feature, taken over the target batch itself, so the metric
    is invariant to a constant offset and to per-dim rescaling of the space.
    """
    p, t = _flat(pred), _flat(target)
    mu = t.mean(dim=0, keepdim=True)
    num = (p - t).pow(2).sum(dim=-1).mean()
    den = (t - mu).pow(2).sum(dim=-1).mean().clamp_min(_EPS)
    return (num / den).item()


def r2(pred: Tensor, target: Tensor) -> float:
    """``1 - NMSE``. 1 perfect, 0 mean-predictor, <0 worse than the mean."""
    return 1.0 - nmse(pred, target)


def cosine_error(pred: Tensor, target: Tensor) -> float:
    """``1 - mean cos(pred, target)``. Isolates global scale + rotation: pure
    direction quality, independent of magnitude."""
    p, t = _flat(pred), _flat(target)
    return (1.0 - F.cosine_similarity(p, t, dim=-1)).mean().item()


def relative_l2(pred: Tensor, target: Tensor) -> float:
    """``mean( ||pred - target|| / ||target|| )`` per token -- error as a
    fraction of the target's own norm."""
    p, t = _flat(pred), _flat(target)
    return ((p - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(_EPS)).mean().item()


def raw_errors(pred: Tensor, target: Tensor) -> dict[str, float]:
    """In the model's own latent units. Tag: raw -- debugging only, never ranked."""
    p, t = _flat(pred), _flat(target)
    return {
        "raw_mse": (p - t).pow(2).mean().item(),
        "raw_l1": (p - t).abs().mean().item(),
        "raw_smooth_l1": F.smooth_l1_loss(p, t).item(),
    }


def skill(nmse_model: float, nmse_reference: float) -> float:
    """``1 - NMSE_model / NMSE_reference``.

    ``> 0`` -- the model beats the reference predictor (which lives in the same
    latent space, so the ratio is portable across models).
    ``= 0`` -- no better than the reference.  ``< 0`` -- worse.
    """
    return 1.0 - nmse_model / max(nmse_reference, _EPS)


def one_step_report(
    pred: Tensor,
    target: Tensor,
    z_prev: Tensor,
    z_prev2: Tensor | None = None,
) -> dict[str, float]:
    """Full Tier-1 block for one aligned ``(pred, target)`` pair.

    ``z_prev``  -- latent each prediction steps from -> persistence baseline
                   (``next == current``).
    ``z_prev2`` -- one latent frame earlier -> constant-velocity baseline
                   (``next == 2*z_prev - z_prev2``); pass ``None`` to skip.

    The 'skill vs untrained twin' score is *not* here: it needs the twin's NMSE
    and is assembled by the runner via :func:`skill`.
    """
    m: dict[str, float] = {
        "nmse": nmse(pred, target),
        "cosine_error": cosine_error(pred, target),
        "relative_l2": relative_l2(pred, target),
        **raw_errors(pred, target),
    }
    m["r2"] = 1.0 - m["nmse"]

    n_persist = nmse(z_prev, target)
    m["nmse_persistence"] = n_persist
    m["skill_vs_persistence"] = skill(m["nmse"], n_persist)

    if z_prev2 is not None:
        n_cv = nmse(2.0 * z_prev - z_prev2, target)
        m["nmse_const_velocity"] = n_cv
        m["skill_vs_const_velocity"] = skill(m["nmse"], n_cv)

    return m
