"""Tier 6 -- downstream planning.  Tag: shared (space-independent).

CEM plans an action sequence to carry the model's own open-loop rollout from a
start context to a goal latent, then scores that plan against the *true* logged
actions -- which did reach that goal. No training, no decoder: the world model
is frozen and CEM is a derivative-free search over actions at inference time.

The CEM cost uses the model's own latent (that is how you would plan with it);
the *scoring* -- planned vs true actions -- is neutral, so the number is
comparable across architectures with different latent spaces.
"""
from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def ar1_noise(shape, rho: float, device, generator=None) -> Tensor:
    """Unit-variance white noise smoothed by an AR(1) filter along dim=1 (time).
    Real robot actions are frame-to-frame correlated; white-noise CEM samples are
    out of distribution for every model and flatten the cost surface."""
    w = torch.randn(shape, device=device, generator=generator)
    out = torch.empty_like(w)
    out[:, 0] = w[:, 0]
    a = (1.0 - rho ** 2) ** 0.5
    for t in range(1, shape[1]):
        out[:, t] = rho * out[:, t - 1] + a * w[:, t]
    return out


def make_rollout_fn(adapter, latents: Tensor, zero_states: Tensor, horizon: int,
                    chunk: int = 64):
    """(cand actions (N, T, A)) -> final rolled-out latent (N, P, D).

    ``latents``: the full (1, T_lat, P, D) encode output for one clip. The
    adapter's ``rollout`` seeds from ``latents[:, :num_context_frames]`` and then
    feeds itself -- the real frames past the seed are never read, so no leak.
    ``zero_states``: the adapter's aligned zero state tensor (only V-JEPA uses it).
    Chunked so a big CEM population doesn't OOM the predictor.
    """
    @torch.no_grad()
    def fn(cand: Tensor) -> Tensor:
        N = cand.shape[0]
        outs = []
        for i in range(0, N, chunk):
            c = cand[i:i + chunk]
            b = c.shape[0]
            rp = adapter.rollout(latents.expand(b, -1, -1, -1),
                                 c, zero_states.expand(b, -1, -1), horizon)   # (b, H, P, D)
            outs.append(rp[:, -1])
        return torch.cat(outs, dim=0)                                         # (N, P, D)

    return fn


@torch.no_grad()
def cem_plan(
    rollout_fn,
    goal: Tensor,                # (P, D)
    n_actions: int,              # length of the action sequence to optimise
    action_dim: int,
    prior_mean: Tensor,          # (A,)
    prior_std: Tensor,           # (A,)
    device: str = "cuda",
    n_samples: int = 128,
    n_iters: int = 3,
    n_elite: int = 16,
    rho: float = 0.6,
    init_std_scale: float = 2.0,
    clip_sigmas: float = 4.0,
) -> Tensor:
    A = action_dim
    pm = prior_mean.to(device).view(1, A)
    ps = prior_std.to(device).view(1, A)
    mu = pm.expand(n_actions, A).clone()
    sigma = (ps * init_std_scale).expand(n_actions, A).clone()
    lo, hi = pm - clip_sigmas * ps, pm + clip_sigmas * ps
    goal = goal.to(device)

    for _ in range(n_iters):
        eps = ar1_noise((n_samples, n_actions, A), rho, device)
        cand = (mu[None] + sigma[None] * eps).clamp(lo, hi)   # (N, T, A)
        final = rollout_fn(cand)                              # (N, P, D)
        cost = (final - goal[None]).flatten(1).pow(2).sum(-1)
        elite = cand[cost.topk(n_elite, largest=False).indices]
        mu = elite.mean(dim=0)
        sigma = elite.std(dim=0).clamp_min(ps * 0.1)
    return mu                                                 # (n_actions, A)


def plan_action_error(planned: Tensor, true: Tensor, prior_mean: Tensor, n_scored: int) -> dict:
    """planned / true: (T, A) in the adapter's aligned action space.

    ``plan_skill_vs_prior`` normalises out the action units: 1 - MSE(plan)/MSE(prior);
    > 0 means the plan beats just emitting the average action, and it is comparable
    across models the way a Tier-1 skill score is.
    """
    k = max(1, min(n_scored, planned.shape[0], true.shape[0]))
    p, t = planned[:k].float().cpu(), true[:k].float().cpu()
    prior = prior_mean.view(1, -1).float().cpu().expand_as(t)
    mse = (p - t).pow(2).mean().item()
    mse_prior = (prior - t).pow(2).mean().item()
    return {
        "plan_action_mse": mse,
        "plan_action_mse_prior": mse_prior,
        "plan_skill_vs_prior": 1.0 - mse / max(mse_prior, _EPS),
        "n_scored": k,
    }


def success_curve(mses: list[float], n_tau: int = 40) -> dict:
    """Empirical CDF of the per-episode planned-action MSE: Success @ tau."""
    import numpy as np

    arr = np.sort(np.asarray(mses, dtype=float))
    if arr.size == 0:
        return {"tau": [], "success": [], "median": float("nan")}
    hi = float(np.quantile(arr, 0.95)) or float(arr.max() or 1.0)
    taus = np.linspace(0.0, hi, n_tau)
    succ = [float((arr < tau).mean()) for tau in taus]
    return {
        "tau": taus.tolist(),
        "success": succ,
        "median": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p75": float(np.quantile(arr, 0.75)),
    }
