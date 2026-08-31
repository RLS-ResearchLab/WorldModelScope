"""Tier 6 planning eval -- CEM goal-reaching over one model's own rollout, scored
against the true logged actions. Frozen model, no decoder.

    .venv/bin/python -m evaluation.plan_eval --model lewm \
        --ckpt checkpoints/lewm_bridge/step_0016000.pt --wandb

start  = first `num_context_frames` real latent frames of a canonical clip
goal   = the model's latent frame nearest canonical frame `--goal-frame`
         (V-JEPA's latent rate is half the canonical rate, so the step count
         differs per model but the physical span is the same)
plan   = CEM action sequence whose predicted rollout lands closest to the goal
score  = MSE(plan, true aligned actions) + skill vs the action prior + Success @ tau

Writes results/<resolved-model>/planning.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from evaluation.common.data_spec import VAL_LOCAL, build_loader
from evaluation.common.metrics.planning import (
    cem_plan,
    make_rollout_fn,
    plan_action_error,
    success_curve,
)
from evaluation.common.registry import build_adapter, resolved_name

_REPO = Path(__file__).resolve().parents[1]


def _displacement(states: torch.Tensor) -> float:
    """L2 move of the EE-position dims (first 3) between first and last frame."""
    s = states[0].float()
    return (s[-1, :3] - s[0, :3]).norm().item()


@torch.no_grad()
def run(
    model: str,
    ckpt: str,
    split: str = VAL_LOCAL,
    max_clips: int = 200,
    min_displacement: float = 0.0,
    goal_frame: int = 10,
    max_horizon: int = 8,
    seed: int = 0,
    device: str = "cuda",
    cem_samples: int = 96,
    cem_iters: int = 3,
    out: str | None = None,
    wandb_run=None,
) -> dict:
    adapter = build_adapter(model, ckpt, device=device)
    A = adapter.action_dim
    print(f"[{adapter.name}] step={adapter.train_step} P={adapter.tokens_per_frame} "
          f"context={adapter.num_context_frames}  CEM {cem_samples}x{cem_iters}")

    # ---- action prior from the aligned actions of a warm-up pass ----
    warm = []
    for i, b in enumerate(build_loader(split=split, max_clips=120, batch_size=8, seed=seed)):
        warm.append(adapter.align_actions(b.actions).reshape(-1, A).cpu())
        if i >= 15:
            break
    aw = torch.cat(warm)
    prior_mean, prior_std = aw.mean(0), aw.std(0).clamp_min(1e-4)
    print(f"  action prior  mean={prior_mean.tolist()}  std={prior_std.tolist()}")

    rows: list[dict] = []
    skipped_still = 0
    n = 0
    t0 = time.perf_counter()

    for batch in build_loader(split=split, max_clips=None, batch_size=1, seed=seed):
        if n >= max_clips:
            break
        if _displacement(batch.states) < min_displacement:
            skipped_still += 1
            continue

        latents = adapter.encode(batch.frames)                       # (1, T_lat, P, D)
        a_true = adapter.align_actions(batch.actions)[0]             # (T_act, A)
        s_zero = adapter.align_states(torch.zeros_like(batch.states))  # (1, T_s, S)

        T_lat = latents.shape[1]
        C = adapter.num_context_frames
        # canonical goal frame -> this model's latent-frame index (V-JEPA halves the rate)
        from evaluation.common.data_spec import CANON_FRAMES

        goal_idx = min(T_lat - 1, max(C, round(goal_frame * T_lat / CANON_FRAMES)))
        horizon = min(max_horizon, goal_idx - C + 1)
        if horizon < 2:
            continue
        n_actions = a_true.shape[0]
        n_scored = min(n_actions, horizon + C - 1)

        goal = latents[0, C - 1 + horizon]                           # (P, D)
        fn = make_rollout_fn(adapter, latents, s_zero, horizon)
        planned = cem_plan(
            fn, goal, n_actions, A, prior_mean, prior_std, device=device,
            n_samples=cem_samples, n_iters=cem_iters,
        )

        err = plan_action_error(planned, a_true.to(device), prior_mean.to(device), n_scored)
        # diagnostic: how close the plan's own rollout gets, in the model's latent
        achieved = fn(planned[None])[0]
        seed_latent = latents[0, C - 1]
        gap0 = (goal - seed_latent).flatten().norm().item()
        err["latent_goal_gap"] = (achieved - goal).flatten().norm().item() / max(gap0, 1e-6)
        err["episode_id"] = int(batch.episode_id[0])
        rows.append(err)

        n += 1
        if n % 25 == 0:
            m = float(np.median([r["plan_action_mse"] for r in rows]))
            sk = float(np.mean([r["plan_skill_vs_prior"] for r in rows]))
            print(f"  {n:4d} plans  median_mse={m:.4f}  mean_skill_vs_prior={sk:+.3f}", flush=True)

    mses = [r["plan_action_mse"] for r in rows]
    skills = [r["plan_skill_vs_prior"] for r in rows]
    curve = success_curve(mses)
    summary = {
        "model": adapter.name,
        "ckpt": str(ckpt),
        "train_step": adapter.train_step,
        "split": split,
        "n_plans": n,
        "skipped_still": skipped_still,
        "min_displacement": min_displacement,
        "goal_frame": goal_frame,
        "max_horizon": max_horizon,
        "seconds": round(time.perf_counter() - t0, 1),
        "cem": {"samples": cem_samples, "iters": cem_iters},
        "planning": {
            "median_plan_action_mse": curve["median"],
            "p25_plan_action_mse": curve.get("p25"),
            "p75_plan_action_mse": curve.get("p75"),
            "mean_plan_skill_vs_prior": float(np.mean(skills)) if skills else float("nan"),
            "frac_beats_prior": float(np.mean([s > 0 for s in skills])) if skills else float("nan"),
            "median_latent_goal_gap": float(np.median([r["latent_goal_gap"] for r in rows])) if rows else float("nan"),
            "success_at_tau": {"tau": curve["tau"], "success": curve["success"]},
        },
        "episodes": rows,
    }

    dest = Path(out) if out else _REPO / "results" / adapter.name / "planning.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {dest}")
    print(json.dumps({k: summary["planning"][k] for k in
                      ("median_plan_action_mse", "mean_plan_skill_vs_prior", "frac_beats_prior")}, indent=2))

    if wandb_run is not None:
        import wandb

        p = summary["planning"]
        for k, v in p.items():
            if isinstance(v, (int, float)):
                wandb_run.summary[f"planning/{k}"] = v
        wandb_run.summary["planning/n_plans"] = n
        if curve["tau"]:
            wandb_run.log({"planning/success_at_tau": wandb.plot.line_series(
                xs=curve["tau"], ys=[curve["success"]], keys=["success"],
                title="planning - Success @ tau", xname="tau (planned-action MSE)")})
            wandb_run.log({"planning/mse_hist": wandb.Histogram(mses)})

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default=VAL_LOCAL)
    p.add_argument("--max-clips", type=int, default=200)
    p.add_argument("--min-displacement", type=float, default=0.0,
                   help="skip clips whose EE moves less than this (m) start->goal")
    p.add_argument("--goal-frame", type=int, default=10,
                   help="canonical frame index used as the goal (mapped per-model)")
    p.add_argument("--max-horizon", type=int, default=8,
                   help="cap on rollout steps to the goal (rollouts drift past this)")
    p.add_argument("--cem-samples", type=int, default=96)
    p.add_argument("--cem-iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="worldmodelscope-eval")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    torch.manual_seed(args.seed)

    run_handle = None
    if args.wandb:
        import wandb

        run_handle = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"{resolved_name(args.model, args.ckpt)}-planning",
            job_type="planning", mode=args.wandb_mode, config=vars(args),
        )
    try:
        run(model=args.model, ckpt=args.ckpt, split=args.split, max_clips=args.max_clips,
            min_displacement=args.min_displacement, goal_frame=args.goal_frame,
            max_horizon=args.max_horizon, seed=args.seed, device=args.device,
            cem_samples=args.cem_samples, cem_iters=args.cem_iters, out=args.out,
            wandb_run=run_handle)
    finally:
        if run_handle is not None:
            url = getattr(run_handle, "url", None)
            run_handle.finish()
            if url:
                print(f"W&B run: {url}")


if __name__ == "__main__":
    main()
