# WorldModelScope evaluation harness

Model-agnostic comparison of the four Bridge world models. Every headline metric
is **dimensionless** (a ratio that cancels the latent's units) or computed in a
**shared space**, so numbers from different latent spaces are comparable. Raw
latent losses are logged for debugging, never used to rank.

## Layout

```
common/
  adapter.py        WorldModelAdapter contract + BaseAdapter + sliding_windows
  data_spec.py      CanonicalClips  -- one resampling, identical 224px frames for every model
  accumulate.py     exact streaming NMSE / covariance (val set too big to hold in memory)
  registry.py       model name -> adapter factory
  runner.py         (model, ckpt) -> results/<model>/step_<N>.json  (+ optional W&B)
  metrics/
    predictive.py   Tier 1  one-step: nmse, r2, cosine_error, relative_l2, skill(...)
    rollout.py      Tier 2  open-loop: drift ratio, usable horizon, compounding, straightness
    geometry.py     Tier 7  effective rank, variance utilisation, isotropy
    action.py       Tier 3  action ablation, reliance, counterfactual, inversion probe
    probe.py        Tier 5  linear probe to BridgeData state, probe transfer
adapters/
  vjepa2_ac.py      V-JEPA 2.1-AC  (wraps evaluation/vjepa_ac/model.py)
  dino_wm.py        DINO-WM -- one class, both encoders (dino_wm_dinov2 / dino_wm_eupe)
  lewm.py           LeWM bridge -- ViT-S/14 from scratch, CLS token only (P=1)
```

## Data

`val[:4295]` -- the 80 BridgeData V2 validation shards in `raw/bridge_dataset/1.0.0`.
5 Hz -> 4 fps, `image_0`, 224px, 16-frame clips, one per episode, seed 0. All
models get the same bytes; each adapter subsamples to its own geometry.

## Run

```bash
# Step 0 -- data loader gate (~30 s)
.venv/bin/python -m evaluation.common.smoke_data

# full val set, all blocks, logged to W&B  (~15-22 min each on an A40)
.venv/bin/python -m evaluation.common.runner --model vjepa2_ac \
    --ckpt checkpoints/vjepa2_1_ac/rls/vjepa2p1_step_016000.pt --wandb
.venv/bin/python -m evaluation.common.runner --model dino_wm \
    --ckpt checkpoints/dino_wm/dinov2_encoder/step_0016000.pt --wandb
.venv/bin/python -m evaluation.common.runner --model dino_wm \
    --ckpt checkpoints/dino_wm/eupe_encoder/step_0016000.pt --wandb
.venv/bin/python -m evaluation.common.runner --model lewm \
    --ckpt checkpoints/lewm_bridge/step_0016000.pt --wandb

# quick check first: --max-clips 240 --no-untrained  (~1 min)
# append to an existing run instead of a new one: --wandb-run-id <id from the run URL>
# model name is always dino_wm; the adapter reads dinov2 vs eupe from the checkpoint
```

### Checkpoint sweep -- the metric-vs-step curves (plan §5)

Cheap tiers (one-step + rollout + geometry) on *every* checkpoint of a model,
~1000 clips, logged to one W&B run "<model>-curves" on a training-step x-axis.
Answers: is matched-step fair, does skill rise with training or only nmse fall,
when does each model peak. V-JEPA has one checkpoint -- nothing to sweep.

```bash
.venv/bin/python -m evaluation.common.sweep --model dino_wm \
    --ckpt-dir checkpoints/dino_wm/dinov2_encoder --wandb        # ~30 min
.venv/bin/python -m evaluation.common.sweep --model dino_wm \
    --ckpt-dir checkpoints/dino_wm/eupe_encoder --wandb          # ~30 min
.venv/bin/python -m evaluation.common.sweep --model lewm \
    --ckpt-dir checkpoints/lewm_bridge --wandb                   # ~17 min

# -> results/<model>/curve.json + a terminal recap table per model
```

### Planning -- Tier 6 (plan §4), the space-independent number

CEM plans an action sequence to carry each model's own rollout from a start
context to a goal frame, then scores the plan against the *true* logged actions.
Frozen model, no decoder, no training. `plan_skill_vs_prior` (1 - MSE(plan) /
MSE(mean action)) is the cross-model figure; `Success @ tau` is the curve.

```bash
.venv/bin/python -m evaluation.plan_eval --model vjepa2_ac \
    --ckpt checkpoints/vjepa2_1_ac/rls/vjepa2p1_step_016000.pt --wandb   # ~10 min
.venv/bin/python -m evaluation.plan_eval --model dino_wm \
    --ckpt checkpoints/dino_wm/dinov2_encoder/step_0016000.pt --wandb    # ~60 min (P=256)
.venv/bin/python -m evaluation.plan_eval --model dino_wm \
    --ckpt checkpoints/dino_wm/eupe_encoder/step_0016000.pt --wandb      # ~50 min
.venv/bin/python -m evaluation.plan_eval --model lewm \
    --ckpt checkpoints/lewm_bridge/step_0016000.pt --wandb              # ~5 min (P=1)

# tune: --cem-samples 96 --cem-iters 3 --max-clips 200 --goal-frame 10 --max-horizon 8
#       --min-displacement 0.03   (skip clips where the arm barely moves)
# -> results/<model>/planning.json + a Success@tau chart per W&B run
```

W&B: project `worldmodelscope-eval`, entity from `$WANDB_ENTITY`.
* `progress/*` -- running metrics, updated every batch (live curves)
* `final/*`    -- every metric, at the end
* `scorecard/*`-- one point at the checkpoint's training step, to line up with training curves

`--wandb-mode offline` dry-runs without the cloud.

Blocks written per checkpoint: `one_step` (Tier 1), `rollout` (Tier 2),
`geometry` (Tier 7), `action` (Tier 3), `probe` (Tier 5), plus the
skill-vs-untrained-twin control. Skip with `--no-actions` / `--no-probe` /
`--no-untrained`. Tiers 4 (pixels) and 6 (planning) need a trained decoder /
planner -- out of scope.
