"""Run the cheap tiers (one-step + rollout + geometry) on *every* checkpoint of a
model and log the metric-vs-step curves to one W&B run.

Why -- a single full eval is a snapshot. The sweep is the film: it shows whether
a model is still improving at 16k or plateaued (is matched-step fair?), whether
`skill` rises with training or only `nmse` falls while the latent collapses
(is the metric being gamed?), when each model peaks (best-vs-best selection),
and -- for frozen-encoder models -- that geometry is constant across checkpoints
(a free harness correctness check).

Cheap by design: tiers 1 + 2 + geometry only (no action / probe), ~1000 clips,
one untrained twin reused across a model's checkpoints.

    .venv/bin/python -m evaluation.common.sweep \
        --model dino_wm --ckpt-dir checkpoints/dino_wm/dinov2_encoder --wandb

Writes results/<resolved-model>/curve.json and, with --wandb, a run
"<resolved-model>-curves" whose curve/* series is on a training-step x-axis.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from evaluation.common import runner
from evaluation.common.data_spec import VAL_LOCAL
from evaluation.common.registry import resolved_name

_REPO = Path(__file__).resolve().parents[2]


def _step_of(p: Path) -> int:
    try:
        return int(p.stem.split("_")[-1])
    except ValueError:
        return -1


def _sweep_name(model: str, ckpt_dir: str) -> str:
    ckpts = sorted((p for p in Path(ckpt_dir).glob("step_*.pt") if _step_of(p) >= 0), key=_step_of)
    return resolved_name(model, str(ckpts[0])) if ckpts else model


def _curve_row(summary: dict) -> dict[str, float]:
    """The handful of numbers that get their own line chart."""
    t1, t2, t7 = summary["one_step"], summary["rollout"], summary["geometry"]
    dr = t2.get("drift_ratio_by_h", {})
    drift_h4 = dr.get(4) or (dr[max(dr)] if dr else float("nan"))
    row = {
        "curve/nmse": t1["nmse"],
        "curve/r2": t1["r2"],
        "curve/cosine_error": t1["cosine_error"],
        "curve/relative_l2": t1["relative_l2"],
        "curve/skill_vs_persistence": t1.get("skill_vs_persistence", float("nan")),
        "curve/skill_vs_const_velocity": t1.get("skill_vs_const_velocity", float("nan")),
        "curve/skill_vs_untrained": t1.get("skill_vs_untrained", float("nan")),
        "curve/rollout_drift_h4": drift_h4,
        "curve/usable_horizon": t2.get("usable_horizon", 0),
        "curve/compounding_rate": t2.get("compounding_rate", float("nan")),
        "curve/straightness_pred": t2.get("straightness_pred", float("nan")),
        "curve/straightness_real": t2.get("straightness_real", float("nan")),
        "curve/effective_rank": t7["effective_rank"],
        "curve/effective_rank_ratio": t7["effective_rank_ratio"],
        "curve/isotropy": t7["isotropy"],
        "curve/variance_utilisation": t7["variance_utilisation"],
    }
    return {k: (float(v) if v is not None else float("nan")) for k, v in row.items()}


def sweep(
    model: str,
    ckpt_dir: str,
    max_clips: int = 1000,
    batch_size: int = 8,
    seed: int = 0,
    device: str = "cuda",
    wandb_run=None,
) -> dict:
    ckpts = sorted((p for p in Path(ckpt_dir).glob("step_*.pt") if _step_of(p) >= 0), key=_step_of)
    if not ckpts:
        raise FileNotFoundError(f"no step_*.pt in {ckpt_dir}")
    print(f"sweeping {len(ckpts)} checkpoints: {[_step_of(p) for p in ckpts]}")

    name = _sweep_name(model, ckpt_dir)
    if wandb_run is not None:
        import wandb

        wandb_run.name = f"{name}-curves"
        wandb.define_metric("curve/step")
        wandb.define_metric("curve/*", step_metric="curve/step")

    tmp_out = Path(tempfile.gettempdir()) / f"sweep_{model}_{os.getpid()}.json"
    rows: list[dict] = []
    t0 = time.perf_counter()

    for i, ckpt in enumerate(ckpts):
        print(f"\n=== [{i + 1}/{len(ckpts)}] {ckpt.name} ===", flush=True)
        summary = runner.run(
            model=model, ckpt=str(ckpt), split=VAL_LOCAL, max_clips=max_clips,
            batch_size=batch_size, seed=seed, device=device,
            untrained=True, actions=False, probe=False,
            out=str(tmp_out), wandb_run=None,
        )
        if summary["model"] != name:      # trust the built adapter if they disagree
            name = summary["model"]
            if wandb_run is not None:
                wandb_run.name = f"{name}-curves"
        rows.append({
            "step": summary["train_step"],
            "n_clips": summary["n_clips"],
            "one_step": summary["one_step"],
            "rollout": summary["rollout"],
            "geometry": summary["geometry"],
        })
        if wandb_run is not None:
            wandb_run.log({"curve/step": summary["train_step"], **_curve_row(summary)})

    tmp_out.unlink(missing_ok=True)

    out = {
        "model": name,
        "split": VAL_LOCAL,
        "max_clips": max_clips,
        "ckpt_dir": str(ckpt_dir),
        "seconds": round(time.perf_counter() - t0, 1),
        "steps": rows,
    }
    dest = _REPO / "results" / name / "curve.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {dest}  ({len(rows)} checkpoints, {out['seconds']}s)")

    # terminal recap: does skill track nmse, or diverge?
    print("\n step   nmse   skill/persist  skill/untrained  eff_rank  isotropy")
    for r in rows:
        t1, t7 = r["one_step"], r["geometry"]
        print(f"  {r['step']:>5}  {t1['nmse']:.3f}      {t1.get('skill_vs_persistence', float('nan')):+.3f}"
              f"          {t1.get('skill_vs_untrained', float('nan')):+.3f}       "
              f"{t7['effective_rank']:.1f}     {t7['isotropy']:.3f}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="registry name: dino_wm | lewm | vjepa2_ac")
    p.add_argument("--ckpt-dir", required=True, help="directory of step_*.pt")
    p.add_argument("--max-clips", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
            name=f"{_sweep_name(args.model, args.ckpt_dir)}-curves",
            job_type="sweep", mode=args.wandb_mode, config=vars(args),
        )
    try:
        sweep(
            model=args.model, ckpt_dir=args.ckpt_dir, max_clips=args.max_clips,
            batch_size=args.batch_size, seed=args.seed, device=args.device,
            wandb_run=run_handle,
        )
    finally:
        if run_handle is not None:
            url = getattr(run_handle, "url", None)
            run_handle.finish()
            if url:
                print(f"W&B run: {url}")


if __name__ == "__main__":
    main()
