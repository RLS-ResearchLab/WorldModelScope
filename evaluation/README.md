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
  dino_wm.py        DINO-WM dinov2 + eupe        (TODO -- step 2)
  lewm.py           LeWM bridge                  (TODO -- step 5)
```

## Data

`val[:4295]` -- the 80 BridgeData V2 validation shards in `raw/bridge_dataset/1.0.0`.
5 Hz -> 4 fps, `image_0`, 224px, 16-frame clips, one per episode, seed 0. All
models get the same bytes; each adapter subsamples to its own geometry.

## Run

```bash
# Step 0 -- data loader gate (~30 s)
.venv/bin/python -m evaluation.common.smoke_data

# Step 1 -- V-JEPA, full val set, logged to W&B  (~15 min on an A40)
.venv/bin/python -m evaluation.common.runner \
    --model vjepa2_ac \
    --ckpt checkpoints/vjepa2_1_ac/rls/vjepa2p1_step_016000.pt \
    --wandb

# quick check first: --max-clips 200 --no-untrained  (~1 min)
```

W&B: project `worldmodelscope-eval`, entity from `$WANDB_ENTITY`. Each run's
`final/*` summary holds every metric; `scorecard/*` is logged at the checkpoint's
training step so it overlays on the training curves. Use `--wandb-mode offline`
to dry-run without the cloud.

Tiers covered by the runner today: 1, 2, 7 + skill-vs-untrained-twin.
Tiers 3 and 5 land in a later step; 4 and 6 need decoders (out of scope).
