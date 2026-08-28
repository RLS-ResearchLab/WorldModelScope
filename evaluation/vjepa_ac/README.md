# V-JEPA 2.1 AC — checkpoint test harness

Self-contained evaluation of the action-conditioned predictor trained in
[`RLS-ResearchLab/VJEPA2P1_ac`](https://github.com/RLS-ResearchLab/VJEPA2P1_ac)
(`scripts/11_train_vjepa21_base_ac.py`) on the 80 BridgeData V2 validation
shards under `raw/bridge_dataset/1.0.0`. Imports nothing from the rest of this
repo; the reference model code is vendored under `_vendor/vjepa2` (pinned commit
in `_vendor/VJEPA2_COMMIT.txt`).

## The model

| Piece | Arch | Trained? | Checkpoint |
| --- | --- | --- | --- |
| Encoder | V-JEPA 2.1 ViT-B/16, tubelet 2, RoPE | **frozen** (stock Meta weights) | `checkpoints/vjepa2_1_base/vjepa2_1_vitb_dist_vitG_384.pt` |
| Predictor | frame-causal AC ViT, dim 768, depth 12 | **yes** — step 16000 | `checkpoints/vjepa2_1_ac/rls/vjepa2p1_step_016000.pt` |

The Google-Drive file holds **only the predictor** (`state_dict` under
`"predictor"`, plus optimizer/step). The frozen encoder is the public V-JEPA 2.1
ViT-B release:

```bash
mkdir -p checkpoints/vjepa2_1_base
curl -L -C - -o checkpoints/vjepa2_1_base/vjepa2_1_vitb_dist_vitG_384.pt \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

## How it works

```
frames (B,16,256,256,3) ── resize 224, [-1,1] ──► video (B,3,16,224,224)
                                                      │
                                    frozen encoder ───┤   tubelet 2 ⇒ 8 latent frames
                                                      ▼
                              z  (B, 8·196, 768)          196 = (224/16)² tokens / frame
   actions a = action[idx[1::2]]  (B,8,7) ──────────────┤
   states  s = state [idx[::2]]   (B,8,7) ──────────────┤   AC predictor (frame-causal)
                                                      ▼
                              z_pred (B, 8·196, 768)
```

Training loss and primary metric (`k = 196`):

```
smooth_l1( z_pred[:, :-k] , z[:, k:] )     # predict the next latent frame
```

Latent-space only — there is no pixel decoder. Reported:

* `smooth_l1` / `l1` / `mse` / `cos_sim` — one-step next-latent error
* `identity_smooth_l1` / `identity_l1` — "next frame == current frame" baseline;
  the trained model must beat this
* `rollout_l1_h1..h7` (`--rollout`) — open-loop: seed with frame 0, feed the
  model its own predictions, latent L1 `h` steps out

## Run

```bash
# from repo root, in the project venv
TF_CPP_MIN_LOG_LEVEL=3 .venv/bin/python -m evaluation.vjepa_ac.evaluate \
    --max-clips 1000 --rollout --wandb
```

* omit `--max-clips` to score all ~3.9k usable val episodes (~8 min on an A40)
* `--wandb` streams running means to project `bridgedata-vjepa21-ac`
  (`job_type=eval`); use `--wandb-mode offline` for a scratch run
* writes `evaluation/vjepa_ac/metrics_step_<step>.json`
* deterministic in `--seed`

Smoke: `--max-clips 24 --wandb-mode offline` (~7 s).

## W&B output

* `progress/*` vs `progress/clips` — live curves that converge as clips accumulate
* `final/*` — scalar summary (Runs table)
* `eval/clip_smooth_l1_hist` — per-clip distribution
* `eval/rollout_curve` — latent L1 vs horizon
* `heldout/next_latent_smooth_l1` at `heldout/step` — one point on the
  training-step axis; add it to the training run's held-out chart to compare

## Files

| File | Role |
| --- | --- |
| `model.py` | `VJEPA2AC` — builds + loads both checkpoints; `encode`, `predict`, `teacher_forced`, `rollout` |
| `bridge_val.py` | `BridgeValClips` — streams `val[:4295]` (= the 80 shards) with training-identical clip sampling |
| `evaluate.py` | CLI, aggregation, W&B, JSON |
| `_vendor/vjepa2/` | pinned `facebookresearch/vjepa2` model code |

## Reference (step 16000, seed 0, 24-clip smoke)

`smooth_l1 ≈ 0.052`, `cos_sim ≈ 0.954`, identity baseline `≈ 0.165`,
rollout `h1 ≈ 0.24 → h7 ≈ 0.36`. Use ≥1000 clips for a citable figure.
