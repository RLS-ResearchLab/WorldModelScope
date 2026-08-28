"""Evaluate a trained V-JEPA 2.1-AC predictor on the 80 BridgeData V2 val shards.

Teacher-forced next-latent error (the training metric
`heldout_next_latent_smooth_l1`), a copy-previous-frame baseline, and an
optional open-loop latent rollout. Streams results to Weights & Biases so the
running means can be watched live.

    .venv/bin/python -m evaluation.vjepa_ac.evaluate \
        --predictor-ckpt checkpoints/vjepa2_1_ac/rls/vjepa2p1_step_016000.pt \
        --max-clips 1000 --rollout --wandb
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.vjepa_ac.bridge_val import BridgeValClips
from evaluation.vjepa_ac.model import VJEPA2AC

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
DEFAULT_ENCODER = _REPO / "checkpoints/vjepa2_1_base/vjepa2_1_vitb_dist_vitG_384.pt"


def _running_mean(agg: dict) -> dict:
    return {k: float(np.mean(v)) for k, v in agg.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictor-ckpt", default="checkpoints/vjepa2_1_ac/rls/vjepa2p1_step_016000.pt")
    p.add_argument("--encoder-ckpt", default=str(DEFAULT_ENCODER))
    p.add_argument("--data-root", default="raw/bridge_dataset/1.0.0")
    p.add_argument("--max-clips", type=int, default=None, help="cap clips scored (~3.9k available)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout", action="store_true", help="also compute open-loop latent rollout L1")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    # --- W&B ---
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="bridgedata-vjepa21-ac")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B team/user; default = WANDB_ENTITY env or your default entity")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model = VJEPA2AC(args.encoder_ckpt, args.predictor_ckpt, device=args.device)
    print(f"loaded predictor step={model.train_step}  params={model.param_counts}  device={args.device}")

    ds = BridgeValClips(args.data_root, max_clips=args.max_clips, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or f"eval-step{model.train_step}",
            job_type="eval",
            mode=args.wandb_mode,
            config={
                "predictor_ckpt": args.predictor_ckpt,
                "encoder_ckpt": args.encoder_ckpt,
                "predictor_step": model.train_step,
                "split": ds.split,
                "batch_size": args.batch_size,
                "max_clips": args.max_clips,
                "seed": args.seed,
                "encoder_params": model.param_counts["encoder"],
                "predictor_params": model.param_counts["predictor"],
            },
        )
        wandb.define_metric("progress/clips")
        wandb.define_metric("progress/*", step_metric="progress/clips")

    agg = defaultdict(list)          # scalar metrics -> list of per-batch means
    clip_smooth_l1: list[float] = []  # every clip, for the histogram
    n = 0
    t0 = time.perf_counter()

    for batch in loader:
        video = model.preprocess_frames(batch["frames"])
        a, s = batch["actions"], batch["states"]

        tf_m = model.teacher_forced(video, a, s)
        clip_smooth_l1.extend(tf_m.pop("clip_smooth_l1"))
        for k, v in tf_m.items():
            agg[k].append(v)
        if args.rollout:
            for k, v in model.rollout(video, a, s).items():
                agg[k].append(v)
        n += video.shape[0]

        run_m = _running_mean(agg)
        print(f"  {n:5d} clips  smooth_l1={run_m['smooth_l1']:.4f}  "
              f"cos={run_m['cos_sim']:.4f}  (baseline {run_m['identity_smooth_l1']:.4f})", flush=True)
        if run:
            wandb.log({"progress/clips": n, **{f"progress/{k}": v for k, v in run_m.items()}})

    final = _running_mean(agg)
    summary = {
        "predictor_ckpt": args.predictor_ckpt,
        "encoder_ckpt": args.encoder_ckpt,
        "predictor_step": model.train_step,
        "split": ds.split,
        "n_clips": n,
        "seconds": round(time.perf_counter() - t0, 1),
        "metrics": final,
    }
    print(json.dumps(summary, indent=2))

    out = Path(args.out) if args.out else _HERE / f"metrics_step_{model.train_step:06d}.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote", out)

    if run:
        import wandb
        for k, v in final.items():
            wandb.summary[f"final/{k}"] = v
        wandb.summary["final/n_clips"] = n
        wandb.log({"eval/clip_smooth_l1_hist": wandb.Histogram(clip_smooth_l1)})
        # single point on the training-step axis; overlay on the training run's
        # held-out curve in the project workspace.
        wandb.define_metric("heldout/step")
        wandb.define_metric("heldout/*", step_metric="heldout/step")
        wandb.log({"heldout/step": model.train_step,
                   "heldout/next_latent_smooth_l1": final["smooth_l1"],
                   "heldout/next_latent_l1": final["l1"],
                   "heldout/cos_sim": final["cos_sim"]})
        if args.rollout:
            hs = sorted(int(k.split("_h")[1]) for k in final if k.startswith("rollout_l1_h"))
            wandb.log({"eval/rollout_curve": wandb.plot.line_series(
                xs=hs, ys=[[final[f"rollout_l1_h{h}"] for h in hs]],
                keys=["latent L1"], title="open-loop latent rollout", xname="horizon (latent frames)")})
        url = getattr(run, "url", None)
        wandb.finish()
        if url:
            print(f"W&B run: {url}")


if __name__ == "__main__":
    main()
