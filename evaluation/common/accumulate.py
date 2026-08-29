"""Streaming accumulators.

The token-level predictions for the full val set do not fit in memory
(V-JEPA: 4295 clips x 7 steps x 196 tokens x 768 dims). These accumulate exact
sufficient statistics chunk by chunk so the ratio metrics come out identical to
computing them on the whole tensor at once.

Pooled (one-vector-per-frame) latents *do* fit, so Tier-7 geometry and the
Tier-5 probe just collect those directly -- see the runner.
"""
from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


class OnlineNMSE:
    """Exact streaming NMSE / R2 / cosine-error / relative-L2 over ``(pred, target)``
    chunks of shape ``(..., D)``.

    NMSE's denominator is the target's total variance about its **global** per-dim
    mean, recovered from running sum / sum-of-squares -- not a per-chunk mean.
    """

    def __init__(self) -> None:
        self.sse = 0.0
        self.sum_t: Tensor | None = None
        self.sum_t2: Tensor | None = None
        self.n = 0
        self.cos_sum = 0.0
        self.rel_sum = 0.0

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor) -> None:
        p = pred.reshape(-1, pred.shape[-1]).double().cpu()
        t = target.reshape(-1, target.shape[-1]).double().cpu()
        if self.sum_t is None:
            d = p.shape[1]
            self.sum_t = torch.zeros(d, dtype=torch.double)
            self.sum_t2 = torch.zeros(d, dtype=torch.double)
        self.sse += (p - t).pow(2).sum().item()
        self.sum_t += t.sum(dim=0)
        self.sum_t2 += t.pow(2).sum(dim=0)
        self.n += p.shape[0]
        self.cos_sum += torch.cosine_similarity(p, t, dim=-1).sum().item()
        self.rel_sum += ((p - t).norm(dim=-1) / t.norm(dim=-1).clamp_min(_EPS)).sum().item()

    @property
    def nmse(self) -> float:
        tot_var = (self.sum_t2 - self.sum_t.pow(2) / max(self.n, 1)).sum().item()
        return self.sse / max(tot_var, _EPS)

    def results(self) -> dict[str, float]:
        nmse = self.nmse
        return {
            "nmse": nmse,
            "r2": 1.0 - nmse,
            "cosine_error": 1.0 - self.cos_sum / max(self.n, 1),
            "relative_l2": self.rel_sum / max(self.n, 1),
        }


class OnlineCovariance:
    """Running mean and covariance of ``(N, D)`` rows, for Tier-7 geometry when the
    pooled latents are still too many to keep (they usually aren't -- this is a
    safety net)."""

    def __init__(self) -> None:
        self.sum: Tensor | None = None
        self.outer: Tensor | None = None
        self.n = 0

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        x = x.reshape(-1, x.shape[-1]).double().cpu()
        if self.sum is None:
            d = x.shape[1]
            self.sum = torch.zeros(d, dtype=torch.double)
            self.outer = torch.zeros(d, d, dtype=torch.double)
        self.sum += x.sum(dim=0)
        self.outer += x.T @ x
        self.n += x.shape[0]

    @property
    def mean(self) -> Tensor:
        return self.sum / max(self.n, 1)

    @property
    def covariance(self) -> Tensor:
        mu = self.mean
        return self.outer / max(self.n, 1) - torch.outer(mu, mu)


class OnlineMean:
    """Running mean of scalars or same-shape tensors."""

    def __init__(self) -> None:
        self.total = 0.0
        self.n = 0

    def update(self, value: float, weight: int = 1) -> None:
        self.total += float(value) * weight
        self.n += weight

    @property
    def value(self) -> float:
        return self.total / max(self.n, 1)
