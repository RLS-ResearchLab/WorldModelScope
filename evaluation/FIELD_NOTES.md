# World-Model Eval — Field Notes

A model-agnostic comparison of four Bridge world models. This file records where the
harness stands, what each metric block measures and how to read its plot, the preliminary
picture from the first runs, and the work still needed to make it a defensible report.

- **Scope:** V-JEPA 2.1-AC · DINO-WM (DINOv2) · DINO-WM (EUPE) · LeWM-bridge
- **Data:** BridgeData V2 `val[:4295]`, the 80 local validation shards
- **Status:** harness live · 3 of 4 adapters · latent tiers only (3, 4, 6 pending)

> Living document — figures are from the first runs and will move as the full sweep and
> LeWM land. Rendered copies: `FIELD_NOTES.pdf`, `FIELD_NOTES.txt` (regenerate with
> `evaluation/render_field_notes.py`).

---

## 1. Why the raw numbers can't be compared

A world model is an encoder `f` (observation → latent) plus a predictor `g` (latent + action
→ next latent). Each of the four models trains **its own latent space**, so each model's
training loss is in its own units, measured against its own encoder. `smooth_l1 = 0.047` for
V-JEPA is not the same quantity as `0.047` for DINO-WM.

Worse, a lower loss can mean a *weaker* model: a smoother, lower-rank, slower-varying latent
is trivially easier to predict, and a fully collapsed latent has zero error.

**The one principle:** every headline number is either **dimensionless** (a ratio that
cancels the latent's units) or computed in a **shared space** (pixels / frozen judge / task
success). Rank only on `skill` scores (model error ÷ a baseline's error, both in the same
space) and shared-space metrics. Raw latent losses are logged for debugging, never used to
rank.

---

## 2. What's built

One canonical clip stream feeds every model the same bytes; each adapter subsamples to its
own geometry. Metric code imports no model. The runner streams exact sufficient statistics
(token-level predictions for the full val set don't fit in memory) and logs live to W&B.

| Component | State | Notes |
|---|---|---|
| Canonical data spec | done | `val[:4295]`, 16 frames @ 4 fps, 224 px, one camera, fixed seed; resized once so every model sees identical pixels |
| V-JEPA 2.1-AC adapter | done | wraps existing model code; 16 frames -> 8 latent frames x 196 tokens x 768; feeds per-frame action + state |
| DINO-WM adapter | done | one class, both encoders (`dino_wm_dinov2` / `dino_wm_eupe`), rebuilt from each checkpoint's embedded config |
| LeWM-bridge adapter | to do | single CLS vector per frame (P = 1); encoder trained from scratch; the integration risk - built last |
| Five metric blocks | done | one-step, rollout, geometry, action, probe - all wired, plus the untrained-twin control |
| Checkpoint sweep + report.py | next | encode-once sweep for metric-vs-step curves; fold all JSON into the ranked scorecard |

Out of scope, both blocked on training something new: **Tier 4** (decode to pixels, LPIPS)
needs V-JEPA + LeWM decoders; **Tier 6** (CEM goal-reaching, the space-independent ranking)
needs the planning loop.

---

## 3. Where each model stands

| Model | Checkpoints | Eval run | Note |
|---|---|---|---|
| `vjepa2_ac` | 16k only | full, 4106 clips | 16k of a planned 80k - every number is *early* |
| `dino_wm_dinov2` | 1k-16k (16) | smoke only (<=200 clips) | adapter verified; full run pending |
| `dino_wm_eupe` | 1k-16k (16) | smoke only (<=200 clips) | adapter verified; full run pending |
| `lewm_bridge` | 1k-12k (12) | not started | caps the matched-step comparison at 12k |

### V-JEPA 2.1-AC — full run, step 16k

| block | metric | value | reading |
|---|---|---|---|
| one-step | `nmse / r2` | 0.233 / 0.767 | ~23% of target variance unexplained |
| one-step | `cosine_error` | 0.041 | direction near-perfect - the gap is scale |
| one-step | `skill_vs_persistence` | 0.742 | predictor adds a lot over "next = current" |
| one-step | `skill_vs_const_velocity` | 0.897 | latent path is not linear (const-vel NMSE 2.26) |
| one-step | `skill_vs_untrained` | 0.936 | training >> a random predictor on the same encoder |
| rollout | `usable_horizon` | 3 | R2 >= 0.5 out to 3 open-loop steps |
| rollout | `drift_ratio @ h7` | 2.45 | error 2.5x the 1-step error, then plateaus |
| action | `action_reliance` | 0.015 | the action input is almost inert at this step |
| action | `action_inversion_r2` | 0.39 | the transition pair still encodes the action linearly |
| probe | `probe_r2_real` | 0.525 | EE position dims probe 0.78-0.85; rotation/gripper 0.13-0.48 |
| probe | `probe_transfer` | 1.04 | predicted latents keep the task info intact |
| geometry | `effective_rank` | 17 / 768 | latent lives in a low-dimensional subspace |

Geometry `isotropy` for V-JEPA still reflects the old (overflow-prone) formula - the re-run
picks up the corrected metric.

### DINO-WM — preliminary smoke samples (24–200 clips, **not** the full run)

| metric | DINOv2 | EUPE | reading |
|---|---|---|---|
| `nmse` | 0.14 | 0.10 | lower raw error than V-JEPA (0.23) |
| `nmse_persistence` | 0.19 | 0.12 | copying the last frame is already most of the answer |
| `skill_vs_persistence` | 0.27 | 0.11 | the predictor adds little on top of that |
| `skill_vs_const_velocity` | 0.68 | 0.61 | linear extrapolation is a decent predictor here |
| `action_reliance` | 0.19 | 0.13 | genuinely uses the action - unlike V-JEPA |
| `counterfactual_divergence` | 0.45 | - | a wrong action moves the prediction ~45% of true motion |
| `usable_horizon` | 13 | - | rollout never drops below R2 0.5 within the clip |
| `effective_rank` | 13 / 384 | 13 / 384 | also low-rank |
| `isotropy` | 0.36 | - | moderately coned (the known DINOv2 anisotropy) |

---

## 4. The five readouts — what each plot means

Every run writes one JSON with five blocks. Each becomes a set of W&B panels: a live
`progress/*` curve that converges as clips accumulate, and a `scorecard/*` point at the
checkpoint's training step.

### one_step — scale-fair next-latent prediction

Feed real context, predict the next latent, score it against the encoder's real output.

- **`nmse` / `r2`** — error normalised by target variance. `r2 = 1 - nmse`: 1 perfect, 0 =
  no better than predicting the mean, <0 worse.
- **`cosine_error`** — direction quality only, independent of magnitude. High `r2` with high
  `cosine_error` = right scale, wrong direction.
- **`skill_vs_persistence`** — `1 - nmse / nmse("next = current")`. > 0 means the predictor
  beats copying the last frame.
- **`skill_vs_const_velocity`** — same against `2*z_t - z_{t-1}`. > 0 means it learned
  something beyond linear motion.
- **`skill_vs_untrained`** — same against a random-weight predictor on the identical frozen
  encoder. This is what *training* added, isolated from the latent space.

> **Reading:** rank on the `skill_*` scores, not `nmse`. A low `nmse` next to a low
> `skill_vs_persistence` means the latent barely moves frame-to-frame — the score is easy,
> not good.

### rollout — open-loop stability

Seed with real context, then feed the model its own predictions. Horizons differ by
architecture (V-JEPA 7, DINO-WM 13), so the comparable metric is the self-normalised one.

- **`nmse_by_h`** — the error-vs-horizon curve. Its *shape* matters, not just the endpoint —
  a plateau is graceful, a straight climb is divergence.
- **`drift_ratio_by_h`** — `nmse(h) / nmse(1)`, always starts at 1.0. Removes "how good is
  step 1" and leaves pure compounding. **The cross-model rollout metric.**
- **`usable_horizon`** — largest h with `r2(h) >= 0.5`. One integer per model.
- **`compounding_rate`** — slope of `log nmse` vs h — the exponential drift constant.
- **`straightness`** — mean turning angle along the trajectory. On the real path: how
  linearly predictable the representation is. On the rollout: whether the model keeps the
  trajectory's shape.

> **Reading:** overlay every model on the `drift_ratio` panel and compare at a fixed horizon
> (say h4). The lowest curve is the most stable world model, regardless of how easy its
> latent is.

### geometry — why a model wins the other blocks

Computed on the encoded real latents only. Does not rank models — explains the ranking.

- **`effective_rank`** — participation ratio of the latent covariance spectrum. Far below
  the latent width = partial collapse; such a latent is trivially easy to predict.
- **`variance_utilisation`** — fraction of dims carrying real variance. Dead dims = wasted
  capacity.
- **`isotropy`** — `1 - ||mean unit-vector||^2`. 1 = tokens point every which way; near 0 =
  confined to a narrow cone (the DINOv2 anisotropy).

> **Reading:** if a model has low `nmse` *and* low `effective_rank`, discount the `nmse` win
> and trust `drift_ratio` and the skill scores instead.

### action — is the conditioning real

Re-run the predictor with shuffled and zeroed actions (the encoder pass is reused).

- **`action_reliance`** — `(nmse_shuffled - nmse_real) / nmse_shuffled`. 0 = ignores the
  action, -> 1 = fully depends on it.
- **`action_effect_vs_zero`** — same against zeroed actions — separates "no action" from
  "wrong action".
- **`counterfactual_divergence`** — how far a wrong action moves the prediction, relative to
  the true one-step motion. ~1 calibrated, <1 under-responsive.
- **`action_inversion_r2`** — can a linear map recover the action from `(z_t, z_{t+1})`.
  High = the transition encodes the action's effect.

> **Reading & a caveat:** if `action_reliance` ~ 0 (V-JEPA at 16k), the rollout curves show
> how well the model predicts *the average plausible future*, not the future given these
> actions — a stable curve there does not imply planning usefulness. Note: V-JEPA also
> receives real per-frame *states*, kept real during the action shuffle, so its number reads
> as "does the commanded action add anything beyond the observed state trajectory".

### probe — do predictions keep the task info

Ridge-fit a linear map from pooled latent to the BridgeData EE-state vector, on an
episode-split.

- **`probe_r2_real`** — how much end-effector state the representation exposes linearly.
- **`probe_transfer`** — `r2(predicted) / r2(real)`. 1 = predictions keep the task info;
  <1 = semantically degraded even where they look plausible.

> **Reading:** a model can have great `nmse` and poor `probe_transfer` — "looks plausible,
> forgot the task". Only meaningful once `probe_r2_real` is well above zero.

---

## 5. The preliminary reading

From the V-JEPA full run and the DINO-WM smoke samples — **directional only**: the DINO-WM
numbers are <=200 clips against V-JEPA's 4106, LeWM is missing, and V-JEPA is 20% trained.

### Rollout NMSE vs horizon (shape contrast is the robust part)

```
NMSE
0.7 |
0.6 |                  V o --- o --- o       V-JEPA: steep, then plateaus
0.5 |.......... R2=0.5 .......................  (usable horizon 3)
0.4 |          V o
0.3 |     V o                    D - o - o - o  DINO-WM DINOv2: gentle,
0.2 |  V o        D - o - o - o                 stays under R2=0.5 all clip
0.1 |  D o
0.0 +----+----+----+----+----+----+----+----+
     1    3    5    7    9   11   13   horizon
```

| h | 1 | 2 | 3 | 4 | 5 | 6 | 7 | ... | 13 |
|---|---|---|---|---|---|---|---|---|---|
| V-JEPA NMSE | 0.25 | 0.39 | 0.48 | 0.55 | 0.62 | 0.63 | 0.61 | - | - |
| DINO-WM NMSE | 0.13 | 0.19 | 0.24 | 0.28 | 0.32 | 0.34 | 0.37 | ... | 0.50 |

### Raw error and skill disagree — and that's the point

- **Raw NMSE** ranks them EUPE (0.10) < DINOv2 (0.14) < V-JEPA (0.23) — "DINO-WM wins".
- **`skill_vs_persistence`** flips it: V-JEPA (0.74) >> DINOv2 (0.27) > EUPE (0.11). In
  DINO-WM's latent, copying the previous frame already scores NMSE 0.12-0.19 — the latents
  barely move — so its predictor only adds 11-27%. V-JEPA's predictor adds 74% on a
  genuinely moving latent (persistence NMSE 0.90).
- **Action grounding** splits the same way in reverse: DINO-WM clearly uses actions
  (reliance 0.13-0.19, counterfactual divergence 0.45); V-JEPA at 16k barely does (0.015).
- **Geometry** is a common caveat: all three are low-rank (13-17 effective dims of 384-768),
  and DINO-WM is also coned (isotropy 0.36). The skill scores and drift ratio are the
  trustworthy metrics on such spaces.

> **Provisional takeaway:** no single number settles it. V-JEPA's predictor does more actual
> predictive work but on a harder latent and without using its actions yet; DINO-WM posts
> lower error largely because its encoder produces a slow, low-rank latent, but it does use
> its actions. The full runs plus LeWM will firm this up.

---

## 6. How to analyse once every run lands

Never read one panel in isolation.

1. **Shape of each rollout curve.** Check `nmse_by_h` at h1 matches the teacher-forced
   `nmse`; read the slope over the first 3 steps; note plateau vs straight climb.
2. **Overlay the drift-ratio curves.** All start at 1.0. Compare at a fixed horizon — lowest
   is the most stable world model, latent difficulty cancelled out.
3. **Cross with one-step skill.** Good skill + flat drift = strong. Good skill + steep drift
   = memorises transitions, can't roll out. Poor skill + flat drift = weak but stable —
   check geometry.
4. **Geometry sanity check.** Low `nmse` plus low `effective_rank` or `isotropy` → lean on
   `drift_ratio` and skill, not raw error.
5. **Action sanity check.** If `action_reliance` ~ 0, the model is doing unconditional
   prediction — flag it for any planning claim.
6. **Write the comparison sentence.** "X predicts Bridge dynamics better than Y by margin M,
   holding after controlling for input, latent units, training budget, and eval-set leakage."

---

## 7. Toward a stronger report and experiment

### Finish the latent comparison

- **LeWM-bridge adapter** — the 4th model. Direct calls into its `JEPA` class, bypassing its
  Hydra eval stack.
- **Checkpoint sweep** with encode-once (the frozen encoder is shared across all 16 DINO-WM
  checkpoints) — a full ladder in ~1 h instead of ~4. Produces the metric-vs-step curves: is
  `skill_vs_persistence` still climbing at 16k, is `effective_rank` collapsing over training.
- **`report.py`** — folds every JSON into the scorecard: 4 models x ~8 dimensionless
  metrics, in a *matched-step* variant (everyone at 12k, LeWM's ceiling) and a *best-vs-best*
  variant, plus a fairness panel stating which controls hold for each pair.

### Make it defensible

- **Multiple seeds** (2-3) with confidence intervals on every headline metric — the smoke
  runs already show ~0.01 wander in `nmse` across samples.
- **Fresh held-out slice** — download shards 80-127 (`val[4295:]`, ~4 GB), a slice no
  model's training touched. Report the `val[:4295]` → `val[4295:]` delta per model; a large
  gap means it saw its eval set too closely.
- **Fix the action-tier confound** for V-JEPA — either add a joint action+state ablation, or
  document clearly that its `action_reliance` is conditional on real states.
- **Resume V-JEPA toward >=40k** if it stays competitive on the dimensionless tiers — the
  16k numbers are preliminary by construction (400 train shards are on disk; the cost is GPU
  time).

### Add the shared-space tiers — the strongest claims

- **Tier 4 — pixels.** Train V-JEPA and LeWM decoders (DINO-WM's exist). Report the
  *LPIPS-penalty*: perceptual error of `decode(prediction)` minus the decoder's own ceiling
  on `decode(encode(real))`. The part attributable to the world model.
- **Frozen judge.** Re-encode every model's predictions with one fixed DINOv2 and score
  there — an absolute "is this latent space better" number.
- **Tier 6 — planning.** CEM goal-reaching over each model's own rollout, scored in the
  judge space or in pixels, never in the model's own latent. *Success @ tau* is the one
  cross-architecture ranking number that survives every objection.

### Report surface

- Headline table — 4 models x ~8 metrics, matched-step and best-vs-best side by side.
- Fairness panel — for each pairwise claim, which of {input, metric, space, maturity, data}
  it controls, with a confidence flag.
- Per-metric training curves; held-out delta per model; a decoded open-loop rollout gallery,
  same clips, side by side.

---

## Pointers

- Harness: `evaluation/common/` (adapter contract, data spec, metrics, runner) ·
  `evaluation/adapters/` (per-model wrappers)
- Results: `results/<model>/step_<N>.json`
- W&B project: `worldmodelscope-eval`
- Run commands: `evaluation/README.md`
