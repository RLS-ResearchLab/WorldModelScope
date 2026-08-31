"""Run a metric set on one (model, checkpoint) over the canonical val stream.

Writes ``results/<model>/step_<N>.json`` and, with ``--wandb``, a W&B run whose
summary holds every headline number, whose ``progress/*`` series updates live as
clips accumulate, and whose ``scorecard/*`` point is logged at the checkpoint's
training step so it lines up with the training curves.

    .venv/bin/python -m evaluation.common.runner \
        --model vjepa2_ac \
        --ckpt checkpoints/vjepa2_1_ac/rls/vjepa2p1_step_016000.pt \
        --wandb

Blocks written: one_step (Tier 1), rollout (Tier 2), geometry (Tier 7),
action (Tier 3), probe (Tier 5), plus the skill-vs-untrained-twin control.
Tiers 4 (pixels) and 6 (planning) need a trained decoder / planner -- out of scope.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import Tensor

from evaluation.common.accumulate import OnlineMean, OnlineNMSE
from evaluation.common.data_spec import CANON_FRAMES, VAL_LOCAL, build_loader
from evaluation.common.metrics.action import action_inversion_r2
from evaluation.common.metrics.geometry import geometry_report
from evaluation.common.metrics.predictive import skill
from evaluation.common.metrics.probe import fit_probe, probe_r2
from evaluation.common.metrics.rollout import path_straightness, stability_from_nmse
from evaluation.common.registry import build_adapter

_REPO = Path(__file__).resolve().parents[2]
_EPS = 1e-8
_INVERSION_ROW_CAP = 40_000        # rows kept for the linear action-inversion probe


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}/"))
        else:
            out[key] = v
    return out


def _probe_block(
    zr: list[Tensor], zp: list[Tensor], st: list[Tensor], ep: list[Tensor], train_frac: float = 0.8
) -> dict:
    """Tier 5: fit a linear probe real-latent -> state on train episodes, then read
    out from held-out real vs predicted latents. Rows are (clip, predicted-frame)."""
    Z_real = torch.cat([x.reshape(-1, x.shape[-1]) for x in zr]).float()
    Z_pred = torch.cat([x.reshape(-1, x.shape[-1]) for x in zp]).float()
    S = torch.cat([x.reshape(-1, x.shape[-1]) for x in st]).float()
    E = torch.cat([e.repeat_interleave(x.shape[1]) for e, x in zip(ep, zr)])

    uniq = torch.unique(E)
    train_eps = set(uniq[: int(train_frac * len(uniq))].tolist())
    m_tr = torch.tensor([e in train_eps for e in E.tolist()])
    if m_tr.all() or (~m_tr).all():                       # tiny run -> no episode split
        m_tr = torch.arange(len(E)) % 5 != 0

    w = fit_probe(Z_real[m_tr], S[m_tr], ridge=10.0)
    r2_real = probe_r2(w, Z_real[~m_tr], S[~m_tr])
    r2_pred = probe_r2(w, Z_pred[~m_tr], S[~m_tr])
    # transfer is only meaningful once the representation itself carries the state
    # linearly; below that floor the ratio is numerically meaningless.
    transfer = (r2_pred["probe_r2_mean"] / r2_real["probe_r2_mean"]
                if r2_real["probe_r2_mean"] > 0.05 else float("nan"))
    return {
        "probe_r2_real": r2_real["probe_r2_mean"],
        "probe_r2_predicted": r2_pred["probe_r2_mean"],
        "probe_transfer": transfer,
        "probe_r2_real_per_dim": r2_real["probe_r2_per_dim"],
        "n_probe_rows": int(len(E)),
    }


@torch.no_grad()
def run(
    model: str,
    ckpt: str,
    split: str = VAL_LOCAL,
    max_clips: int | None = None,
    batch_size: int = 8,
    seed: int = 0,
    rollout_horizon: int | None = None,
    device: str = "cuda",
    untrained: bool = True,
    actions: bool = True,
    probe: bool = True,
    out: str | None = None,
    wandb_run=None,
    wandb_name: str | None = None,
) -> dict:
    adapter = build_adapter(model, ckpt, device=device)
    twin = adapter.build_untrained() if untrained else None
    ac_on = actions and adapter.is_action_conditioned
    print(f"[{adapter.name}] step={adapter.train_step} "
          f"P={adapter.tokens_per_frame} D={adapter.latent_dim} "
          f"context={adapter.num_context_frames}  device={device}  "
          f"action_tier={ac_on} probe_tier={probe}")

    loader = build_loader(split=split, max_clips=max_clips, batch_size=batch_size, seed=seed)

    if wandb_run is not None:
        import wandb

        # name/group from the *resolved* adapter (dino_wm_dinov2 vs dino_wm_eupe),
        # not the CLI --model string which is the same for both encoders.
        if wandb_name is None and not wandb_run.resumed:
            wandb_run.name = f"{adapter.name}-step{adapter.train_step:06d}"
        wandb_run.config.update(
            {"resolved_model": adapter.name, "resolved_step": adapter.train_step,
             "encoder_fingerprint": getattr(adapter, "encoder_fingerprint", None)},
            allow_val_change=True,
        )
        wandb.define_metric("progress/clips")
        wandb.define_metric("progress/*", step_metric="progress/clips")

    m_model, m_persist, m_cv = OnlineNMSE(), OnlineNMSE(), OnlineNMSE()
    m_untrained = OnlineNMSE() if twin else None
    m_shuf = OnlineNMSE() if ac_on else None
    m_zero = OnlineNMSE() if ac_on else None
    cf_div = OnlineMean() if ac_on else None
    roll_nmse: dict[int, OnlineNMSE] = defaultdict(OnlineNMSE)
    straight_pred, straight_real = OnlineMean(), OnlineMean()

    pooled_latents: list[Tensor] = []                          # (B, T, D) -> Tier 7
    inv_zt, inv_zn, inv_a = [], [], []                         # -> action-inversion probe
    pr_zr, pr_zp, pr_st, pr_ep = [], [], [], []                # -> Tier 5 probe

    n_clips, t0 = 0, time.perf_counter()
    for batch in loader:
        latents = adapter.encode(batch.frames)                 # (B, T, P, D)
        a = adapter.align_actions(batch.actions)
        s = adapter.align_states(batch.states)
        B, T, P, D = latents.shape

        tf = adapter.teacher_forced(latents, a, s)
        m_model.update(tf.pred, tf.target)
        m_persist.update(tf.z_prev, tf.target)
        if tf.z_prev2 is not None:
            m_cv.update(2.0 * tf.z_prev - tf.z_prev2, tf.target)
        if twin:
            m_untrained.update(twin.teacher_forced(latents, a, s).pred, tf.target)

        K, ts = tf.target.shape[1], tf.target_start

        # ---- Tier 3: action grounding (encoder pass reused) ----
        if ac_on:
            pred_shuf = adapter.teacher_forced(latents, torch.roll(a, 1, dims=0), s).pred
            pred_zero = adapter.teacher_forced(latents, torch.zeros_like(a), s).pred
            m_shuf.update(pred_shuf, tf.target)
            m_zero.update(pred_zero, tf.target)
            dp = (tf.pred - pred_shuf).flatten(1).norm(dim=-1)
            dr = (tf.target - tf.z_prev).flatten(1).norm(dim=-1).clamp_min(_EPS)
            cf_div.update((dp / dr).mean().item(), B)
            if sum(x.shape[0] for x in inv_a) < _INVERSION_ROW_CAP:
                inv_zt.append(tf.z_prev.mean(2).reshape(-1, D).cpu())
                inv_zn.append(tf.target.mean(2).reshape(-1, D).cpu())
                inv_a.append(a[:, ts - 1: ts - 1 + K].reshape(-1, a.shape[-1]).cpu())

        # ---- Tier 2: rollout ----
        C = adapter.num_context_frames
        H = rollout_horizon or (T - C)
        rp = adapter.rollout(latents, a, s, H)                 # (B, h, P, D)
        rr = latents[:, C:C + rp.shape[1]]
        for h in range(rp.shape[1]):
            roll_nmse[h + 1].update(rp[:, h], rr[:, h])
        if rp.shape[1] >= 3:
            straight_pred.update(path_straightness(rp), B)
            straight_real.update(path_straightness(rr), B)

        # ---- Tier 7 + Tier 5 collection (small, kept in full) ----
        pooled_latents.append(latents.mean(dim=2).reshape(-1, D).cpu())
        if probe:
            # real BridgeData EE state at the canonical frame each predicted latent
            # frame maps to (V-JEPA's latent rate is half the canonical rate).
            # adapter.align_states zeroes the state for models that don't consume
            # it -- that's a predictor input, never the probe target.
            canon = [min(CANON_FRAMES - 1, round((ts + i) * CANON_FRAMES / T)) for i in range(K)]
            pr_zr.append(tf.target.mean(2).cpu())             # real latent at predicted frames
            pr_zp.append(tf.pred.mean(2).cpu())               # predicted latent
            pr_st.append(batch.states[:, canon].cpu())        # real state at those frames
            pr_ep.append(batch.episode_id.clone())

        n_clips += B
        r = m_model.results()
        line = (f"  {n_clips:5d} clips  nmse={r['nmse']:.4f} r2={r['r2']:.4f} "
                f"cos_err={r['cosine_error']:.4f}")
        if ac_on:
            line += f"  a_reliance={(m_shuf.nmse - r['nmse']) / max(m_shuf.nmse, _EPS):.3f}"
        print(line, flush=True)

        if wandb_run is not None:
            prog = {
                "progress/clips": n_clips,
                "progress/nmse": r["nmse"],
                "progress/r2": r["r2"],
                "progress/cosine_error": r["cosine_error"],
                "progress/relative_l2": r["relative_l2"],
                "progress/skill_vs_persistence": skill(r["nmse"], m_persist.nmse),
            }
            if m_untrained is not None and m_untrained.n:
                prog["progress/skill_vs_untrained"] = skill(r["nmse"], m_untrained.nmse)
            if ac_on:
                prog["progress/action_reliance"] = (m_shuf.nmse - r["nmse"]) / max(m_shuf.nmse, _EPS)
                prog["progress/action_effect_vs_zero"] = (m_zero.nmse - r["nmse"]) / max(m_zero.nmse, _EPS)
                prog["progress/counterfactual_divergence"] = cf_div.value
            if roll_nmse:
                prog["progress/rollout_nmse_h1"] = roll_nmse[min(roll_nmse)].nmse
                prog["progress/rollout_nmse_hlast"] = roll_nmse[max(roll_nmse)].nmse
            wandb_run.log(prog)

    # ---- assemble ----
    t1 = m_model.results()
    t1["nmse_persistence"] = m_persist.nmse
    t1["skill_vs_persistence"] = skill(t1["nmse"], m_persist.nmse)
    if m_cv.n:
        t1["nmse_const_velocity"] = m_cv.nmse
        t1["skill_vs_const_velocity"] = skill(t1["nmse"], m_cv.nmse)
    if m_untrained:
        t1["nmse_untrained_twin"] = m_untrained.nmse
        t1["skill_vs_untrained"] = skill(t1["nmse"], m_untrained.nmse)

    t2 = stability_from_nmse({h: acc.nmse for h, acc in sorted(roll_nmse.items())})
    if straight_pred.n:
        t2["straightness_pred"] = straight_pred.value
        t2["straightness_real"] = straight_real.value

    t7 = geometry_report(torch.cat(pooled_latents, dim=0))

    summary = {
        "model": adapter.name,
        "ckpt": str(ckpt),
        "train_step": adapter.train_step,
        "split": split,
        "n_clips": n_clips,
        "seconds": round(time.perf_counter() - t0, 1),
        "encoder_fingerprint": getattr(adapter, "encoder_fingerprint", None),
        "one_step": t1,
        "rollout": t2,
        "geometry": t7,
    }
    if ac_on:
        nmse_real = m_model.nmse
        summary["action"] = {
            "nmse_real_actions": nmse_real,
            "nmse_shuffled_actions": m_shuf.nmse,
            "nmse_zero_actions": m_zero.nmse,
            "action_reliance": (m_shuf.nmse - nmse_real) / max(m_shuf.nmse, _EPS),
            "action_effect_vs_zero": (m_zero.nmse - nmse_real) / max(m_zero.nmse, _EPS),
            "counterfactual_divergence": cf_div.value,
            "action_inversion_r2": action_inversion_r2(
                torch.cat(inv_zt), torch.cat(inv_zn), torch.cat(inv_a)
            ),
        }
    if probe:
        summary["probe"] = _probe_block(pr_zr, pr_zp, pr_st, pr_ep)

    out_path = Path(out) if out else _REPO / "results" / adapter.name / f"step_{adapter.train_step:06d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_path}")
    _head = {k: t1[k] for k in ("nmse", "r2", "cosine_error", "skill_vs_persistence",
                                "skill_vs_untrained") if k in t1}
    if "action" in summary:
        _head["action_reliance"] = summary["action"]["action_reliance"]
    if "probe" in summary:
        _head["probe_transfer"] = summary["probe"]["probe_transfer"]
    print(json.dumps(_head, indent=2))

    if wandb_run is not None:
        blocks = {"one_step": t1, "rollout": t2, "geometry": t7}
        blocks.update({k: summary[k] for k in ("action", "probe") if k in summary})
        flat = _flatten(blocks)
        for k, v in flat.items():
            if isinstance(v, (int, float)):
                wandb_run.summary[f"final/{k}"] = v
        wandb_run.summary["final/n_clips"] = n_clips
        import wandb

        wandb.define_metric("scorecard/step")
        wandb.define_metric("scorecard/*", step_metric="scorecard/step")
        scal = {f"scorecard/{k}": v for k, v in flat.items() if isinstance(v, (int, float))}
        wandb_run.log({"scorecard/step": adapter.train_step, **scal})

        # rollout decay as a single line chart (x = horizon) instead of 7 one-point panels
        hs = sorted(t2["nmse_by_h"])
        wandb_run.log({
            "rollout_curve/nmse_vs_horizon": wandb.plot.line_series(
                xs=hs, ys=[[t2["nmse_by_h"][h] for h in hs]], keys=["NMSE"],
                title="open-loop rollout - NMSE vs horizon", xname="steps ahead"),
            "rollout_curve/drift_vs_horizon": wandb.plot.line_series(
                xs=hs, ys=[[t2["drift_ratio_by_h"][h] for h in hs]], keys=["drift ratio"],
                title="open-loop rollout - drift ratio vs horizon", xname="steps ahead"),
        })

    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="registry name: vjepa2_ac | dino_wm | lewm")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default=VAL_LOCAL)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rollout-horizon", type=int, default=None)
    p.add_argument("--no-untrained", action="store_true", help="skip the untrained-twin control")
    p.add_argument("--no-actions", action="store_true", help="skip Tier 3 (action grounding)")
    p.add_argument("--no-probe", action="store_true", help="skip Tier 5 (state probe)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="worldmodelscope-eval")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-run-id", default=None,
                   help="append to this existing W&B run instead of creating a new one")
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = p.parse_args()

    torch.manual_seed(args.seed)

    run_handle = None
    if args.wandb:
        import wandb

        init_kw = dict(
            project=args.wandb_project, entity=args.wandb_entity,
            job_type="eval", mode=args.wandb_mode, config=vars(args),
        )
        if args.wandb_run_id:
            run_handle = wandb.init(id=args.wandb_run_id, resume="must", **init_kw)
        else:
            from evaluation.common.registry import resolved_name

            provisional = args.wandb_name or f"{resolved_name(args.model, args.ckpt)}-step"
            run_handle = wandb.init(name=provisional, **init_kw)

    try:
        run(
            model=args.model, ckpt=args.ckpt, split=args.split, max_clips=args.max_clips,
            batch_size=args.batch_size, seed=args.seed, rollout_horizon=args.rollout_horizon,
            wandb_name=args.wandb_name,
            device=args.device, untrained=not args.no_untrained,
            actions=not args.no_actions, probe=not args.no_probe, out=args.out,
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
