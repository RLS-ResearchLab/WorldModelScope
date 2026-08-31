# Evaluating Four World Models on BridgeData V2

## Executive summary

A **world model** learns to predict how a scene will change: given a short history of
frames and the action a robot is about to take, it predicts the next state. We compare four
such models trained on the BridgeData V2 robot-manipulation dataset. They are built very
differently and, critically, **each one predicts in its own internal representation** ("latent
space"), which makes their training losses incomparable. This report measures all four on a
single, fair footing: every headline number is either **dimensionless** (a ratio that cancels
the representation's units) or computed in a **shared space**.

The short version of the result:

- **V-JEPA 2.1-AC** is the only model whose predictor has clearly learned the *dynamics* of the
  scene, but it barely uses the robot's action and it is only 20 % through its planned training.
- **DINO-WM** (in two encoder variants, DINOv2 and EUPE) has the lowest raw error, but largely
  because its frozen representation changes very little between frames; its predictor adds a
  real but modest improvement, and it is the only model that genuinely responds to the action.
- **LeWM-bridge** has a near-perfect-looking raw error that is an artefact: in its
  representation, predicting "nothing changes" is already almost optimal, and its predictor is
  in fact slightly *worse* than that trivial baseline.

## What a world model is, and why comparison is hard

Every model here has two parts:

- an **encoder** *f* that turns an image (or a short video) into a vector or grid of vectors,
  its **latent** *z*;
- a **predictor** *g* that takes the latents of the past frames plus the action *a* and
  predicts the **next latent** *ẑ*.

Training minimises the distance between the predicted next latent *ẑ* and the true next latent
*z*, **measured by that model's own encoder in that model's own units**. A loss of 0.047 for
one model is not the same quantity as 0.047 for another. Worse, a model can lower its loss for
the wrong reason: if the encoder learns a representation that changes very little from frame to
frame, the prediction task becomes trivial and the loss collapses even though nothing about
the world's dynamics was learned.

The evaluation therefore never ranks on a raw loss. It ranks on:

- **skill scores** — the model's error *divided by* a simple baseline's error, both computed
  in the same latent space, so the units cancel;
- **shared-space metrics** — quantities computed in pixels, in a fixed third-party encoder, or
  as task success, which do not depend on the model's own representation at all.

## The four models

| Model | Encoder | Latent per frame | Action | Checkpoints used |
|---|---|---|---|---|
| **V-JEPA 2.1-AC** | frozen V-JEPA 2.1 video transformer (ViT-B) | 8 frames × 196 tokens × 768 | 7-D | step 16 000 (of a planned 80 000) |
| **DINO-WM · DINOv2** | frozen DINOv2 image transformer | ~64 tokens × 384 | 7-D | step 16 000 |
| **DINO-WM · EUPE** | frozen EUPE image transformer | ~196 tokens × 384 | 7-D | step 16 000 |
| **LeWM-bridge** | ViT-S/14 **trained from scratch**, single CLS vector | 1 vector × 384 | 7-D | step 16 000 (best is ~12 000) |

"DINO-WM · DINOv2" and "DINO-WM · EUPE" share the same predictor architecture and differ only
in the frozen image encoder. They are treated as two separate models throughout.

## How the comparison is kept fair

| Level | What it controls | Mechanism in this report | Status |
|---|---|---|---|
| **A — Inputs** | frames, resolution, camera, frame rate, random seed | one canonical clip stream feeds all four models the identical 16-frame, 224-px clips | **yes** |
| **B — Metric** | the representation's units, and "easy latent" advantages | rank only on skill scores and shared-space metrics | **yes** |
| **C — Space** | absolute "is this representation better" claims | pixel reconstruction, a frozen judge encoder, planning success | not in this report |
| **D — Maturity** | training budget, cherry-picking of checkpoints | all models reported at the same step (16 000); training curves show whether that is fair | partial |
| **E — Data** | the eval set being too close to training | a fresh, never-trained-on slice of BridgeData | not in this report |

Levels A and B hold fully. Level D is reported with a caveat (LeWM's best checkpoint is earlier
than 16 000). Levels C and E are the subject of the next phase of work.

## Notation used throughout

| Symbol | Meaning |
|---|---|
| z_t | the true latent of frame t (the encoder's output) |
| ẑ_t | the predictor's estimate of z_t |
| z̄ | the mean latent over the whole evaluation set |
| a_t | the 7-D robot action taken at frame t |
| **teacher-forced** | the predictor is given the *real* past latents and predicts one step ahead |
| **open-loop / rollout** | the predictor is fed *its own* previous predictions and runs several steps ahead |
| *h* | the rollout horizon — how many steps have been predicted autoregressively |
| ‖·‖ | Euclidean (L2) norm; ‖·‖² its square |
| ⟨·,·⟩ | inner (dot) product |

The evaluation set is the 80 held-out BridgeData V2 validation shards (`val[:4295]`), 4 106
clips, one clip per episode, fixed seed. Each clip is 16 frames sampled at 4 frames per
second, resized to 224 × 224, from a single camera.

---

## Headline numbers

All four models, evaluated at training step 16 000 on the 4 106-clip held-out set. ★ marks the metrics the comparison is ranked on.

| Metric | V-JEPA | DINOv2 | EUPE | LeWM | Better |
|---| ---: | ---: | ---: | ---: |:--:|
| Skill vs. persistence ★ | 0.742 | 0.256 | 0.115 | -0.136 | higher |
| Skill vs. untrained twin ★ | 0.936 | 0.802 | 0.988 | 0.995 | higher |
| Drift ratio at h = 4 ★ | 2.180 | 2.130 | 2.220 | 6.953 | lower |
| Usable horizon | 3 | 13 | 13 | 13 | higher |
| Rollout straightness / real ★ | 0.939 | 0.646 | 0.562 | 0.240 | ≈ 1 |
| Action reliance ★ | 0.015 | 0.219 | 0.163 | 0.034 | higher |
| Action-inversion R² | 0.401 | 0.113 | 0.101 | 0.0010 | higher |

## The metrics in full

For each metric: its definition, its formula with every symbol spelled out, what it tells a reader, how to read the number, and the result for each model with an interpretation.


### Tier 1 — One-step prediction accuracy

#### Normalised mean-squared error (NMSE)

**Definition.** Teacher-forced: the predictor is given the real past latents and predicts the next one; NMSE is the squared prediction error divided by the natural spread of the target.

**Formula.**

```
NMSE  =  ( Σ_t ‖ ẑ_t − z_t ‖² )  /  ( Σ_t ‖ z_t − z̄ ‖² )
```

The numerator is the model's total squared error. The denominator is the error of the *best constant predictor* — one that ignores the input and always outputs the average latent z̄ — so it equals the total variance of the target.

**What it tells you.** It answers: is the predictor doing anything at all? NMSE = 1 means the model is no better than always guessing the average; NMSE = 0 is perfect. It is **not comparable between models** because each z lives in a different space with a different natural spread.

**Reading.** 0 perfect · 1 = as good as guessing the mean · > 1 = worse than guessing the mean.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.233 | 0.113 | 0.089 | 0.005 |

The four NMSEs span two orders of magnitude — 0.005 for LeWM to 0.23 for V-JEPA — and this spread is almost entirely an artefact of how much each representation moves between frames, not of predictor quality. This is precisely why the report does not rank on it. The numbers are shown so the reader can see the problem the skill scores solve.

#### Coefficient of determination (R²)

**Definition.** A rescaling of NMSE onto the usual "fraction of variance explained" axis.

**Formula.**

```
R²  =  1 − NMSE
```
**What it tells you.** R² = 1 is a perfect predictor, R² = 0 is the constant-mean predictor, R² < 0 is worse than that. Same caveat as NMSE: not comparable across models.

**Reading.** 1 perfect · 0 mean-predictor · < 0 worse than the mean.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.767 | 0.887 | 0.911 | 0.995 |

All four sit above 0.76, and LeWM reaches 0.995 — but, again, on targets of wildly different difficulty. Read it alongside the skill scores below, never alone.

#### Cosine error

**Definition.** One minus the average cosine similarity between the predicted and true next latent — a measure of *direction* only, with the magnitude divided out.

**Formula.**

```
cosine error  =  1  −  mean_t  ⟨ ẑ_t , z_t ⟩ / ( ‖ ẑ_t ‖ · ‖ z_t ‖ )
```

⟨ẑ, z⟩ / (‖ẑ‖‖z‖) is the cosine of the angle between the two vectors: 1 if they point the same way, 0 if orthogonal, −1 if opposite.

**What it tells you.** It separates two kinds of error. If the cosine error is small but NMSE is still large, the predictor points in the right direction but gets the *length* wrong (a scale error). If the cosine error is large, the predictor is pointing the wrong way entirely.

**Reading.** 0 = perfect direction · larger = worse direction. Unlike NMSE it is roughly comparable across models because it is scale-free by construction.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.041 | 0.051 | 0.026 | 0.003 |

Every model has a small cosine error (0.003–0.05), so all four get the *direction* of the next latent almost right. For V-JEPA the cosine error is 0.04 while its NMSE is 0.23 — the gap is a scale error, not a wrong prediction. The DINO-WM · DINOv2 cosine error (0.05) is the largest, consistent with its being the least converged of the DINO pair.

#### Relative L2 error

**Definition.** The prediction error as a fraction of the target's own length, averaged per sample.

**Formula.**

```
relative L2  =  mean_t  ‖ ẑ_t − z_t ‖ / ‖ z_t ‖
```
**What it tells you.** A robustness cross-check on NMSE: NMSE squares the error and normalises by a single global variance, so a few large targets can dominate it; relative L2 normalises each sample by its own norm and does not square, so it is less sensitive to outliers.

**Reading.** 0 = perfect · smaller is better. Not comparable across models (still in latent units).

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.268 | 0.255 | 0.201 | 0.056 |

The relative L2 ordering matches NMSE (LeWM lowest at 0.06, V-JEPA highest at 0.27), confirming that no single outlier clip is driving the NMSE picture.

#### Skill vs. persistence  ★

**Definition.** How much the predictor beats the trivial rule "the next latent equals the current latent." This is the single most important number for whether the model learned dynamics.

**Formula.**

```
skill  =  1  −  NMSE(model)  /  NMSE(persistence)
NMSE(persistence)  =  ( Σ_t ‖ z_{t−1} − z_t ‖² )  /  ( Σ_t ‖ z_t − z̄ ‖² )
```

The *persistence* predictor outputs the previous latent unchanged — it assumes the world does not move. Both NMSEs are computed in the same latent space, so their ratio is dimensionless and **comparable across models**.

**What it tells you.** Beating persistence requires predicting *change*: motion of the arm, of objects, of the scene. A model that has only learned "the next frame looks like this frame" scores 0. Only a model that anticipates motion scores above 0.

**Reading.** > 0 the predictor beats doing nothing · ≈ 0 it is essentially an identity map · < 0 it is *worse* than doing nothing.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.742 | 0.256 | 0.115 | -0.136 |

This is where the models separate sharply. **V-JEPA scores 0.74** — its predictor's error is only a quarter of persistence's, so it has genuinely learned to predict how the scene moves. **DINO-WM adds 0.12–0.26**: a real improvement, but persistence is already a strong baseline in its slow-moving representation, so there is little headroom. **LeWM scores −0.14** — its predictor is measurably worse than simply repeating the last latent. In LeWM's representation the frame-to-frame change is so small (see NMSE of persistence = 0.004 in the appendix) that persistence is near-optimal and the learned predictor only adds noise.

#### Skill vs. constant-velocity

**Definition.** How much the predictor beats linear extrapolation of the last two latents.

**Formula.**

```
skill  =  1  −  NMSE(model)  /  NMSE(constant-velocity)
constant-velocity prediction:  ẑ_t  =  2 z_{t−1}  −  z_{t−2}
```

The constant-velocity baseline assumes the latent keeps moving in a straight line at constant speed — the next-simplest model of motion after persistence.

**What it tells you.** Beating it means the predictor has learned something about *how* motion changes — acceleration, direction changes, contact events — not just that motion continues.

**Reading.** > 0 the model learned beyond straight-line motion.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.897 | 0.673 | 0.595 | 0.249 |

All four beat constant velocity (0.25–0.90), so every predictor has learned at least some non-linear structure. V-JEPA's 0.90 is the strongest; for it, linear extrapolation in the latent is actually *worse than the mean* (NMSE 2.26, appendix), meaning its latent trajectory is strongly curved.

#### Skill vs. untrained twin  ★

**Definition.** How much *training the predictor* helped, with the effect of an easy or hard representation removed.

**Formula.**

```
skill  =  1  −  NMSE(model)  /  NMSE(untrained twin)
```

The *untrained twin* is the exact same architecture — same frozen encoder, same predictor network — but with the predictor's weights left at their random initialisation. It is evaluated on the identical clips.

**What it tells you.** Persistence and constant-velocity are hand-written baselines; the untrained twin is the *model itself before learning*. A high score means training moved the predictor far from random. Because the encoder is identical, this isolates the contribution of predictor training from the difficulty of the latent space.

**Reading.** ≈ 1 training did almost all the work · ≈ 0 a random predictor on this encoder does as well.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.936 | 0.802 | 0.988 | 0.995 |

All four score ≥ 0.80, so in every case predictor training clearly mattered relative to random weights. Note that this can be high even when skill vs. persistence is low or negative (LeWM: 0.995 here, −0.14 there): training moved LeWM's predictor a long way from random, it just moved it to a place that is still worse than the persistence rule.


### Tier 2 — Open-loop rollout stability

#### Rollout NMSE at horizon h, and the drift ratio  ★

**Definition.** When the model is run open-loop — fed its own predictions instead of the real frames — errors accumulate. The drift ratio measures how fast.

**Formula.**

```
NMSE(h)  =  the Tier-1 NMSE, but ẑ_{t+h} comes from h autoregressive steps
drift ratio(h)  =  NMSE(h)  /  NMSE(1)
```

NMSE(1) is one step from real context (close to the teacher-forced NMSE). NMSE(h) for h > 1 feeds the model's own output back in h − 1 times.

**What it tells you.** Dividing by NMSE(1) removes "how good is a single step" and leaves only the *compounding* of error. Because it is a ratio of two NMSEs in the same space, the drift ratio is **directly comparable across models** — it is the cleanest rollout number. A flat drift ratio means a stable model you can plan with; a steep one means the model diverges.

**Reading.** 1.0 = error does not grow with horizon · larger = the model's own mistakes feed back and amplify. Reported here at h = 4.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 2.180 | 2.130 | 2.220 | 6.953 |

V-JEPA (2.18), DINOv2 (2.13) and EUPE (2.22) are close: after four open-loop steps their error is roughly doubled — moderate, controlled drift. **LeWM's drift ratio is 6.95** — after four steps its error is seven times the one-step error. Its rollout is unusable more than two or three steps out, which matters for any downstream planning use.

#### Usable horizon

**Definition.** The furthest the model can be rolled forward before it explains less than half the variance of the true trajectory.

**Formula.**

```
usable horizon  =  max { h  :  R²(h) ≥ 0.5 }
R²(h)  =  1 − NMSE(h)
```
**What it tells you.** A single integer summarising the rollout curve. It is capped by the clip length — 7 predicted steps for V-JEPA (its encoder produces 8 latent frames from 16 images), 13 for DINO-WM and LeWM — so it should be read together with the drift ratio, which is not capped.

**Reading.** Higher is better; the maximum is the number of predictable frames in the clip.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 3 | 13 | 13 | 13 |

DINO-WM and LeWM both reach the cap of 13: in *absolute* terms their rollout NMSE never crosses 0.5 within the clip. V-JEPA's is 3. But note the drift ratio tells a different story for LeWM — its *absolute* rollout error stays low only because its latent barely moves; *relative* to its own one-step error it diverges fastest of the four.

#### Compounding rate

**Definition.** The exponential rate at which rollout error grows per step.

**Formula.**

```
fit  log NMSE(h)  ≈  α  +  β·h   by least squares;   compounding rate  =  β
```

If error grew purely exponentially, NMSE(h) = NMSE(1)·e^{β(h−1)} and β would be its growth constant.

**What it tells you.** A second, slope-based view of the same rollout curve as the drift ratio. Larger β means faster blow-up.

**Reading.** Smaller is better; 0 would mean no growth.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.140 | 0.096 | 0.099 | 0.221 |

V-JEPA 0.14, DINO-WM ~0.10, LeWM 0.22 — the same ordering as the drift ratio, with LeWM compounding fastest.

#### Rollout path straightness, relative to the real path

**Definition.** Whether the model's rolled-out trajectory has the same shape as the true trajectory, or a smoothed / distorted one.

**Formula.**

```
straightness(sequence)  =  mean_t  arccos( ⟨ Δz_t , Δz_{t+1} ⟩ / ( ‖Δz_t‖ · ‖Δz_{t+1}‖ ) )
   where  Δz_t  =  z_{t+1} − z_t     (the step vector at time t)
reported value  =  straightness(rollout)  /  straightness(real trajectory)
```

Δz_t is the direction the latent moves from frame t to t+1. The turning angle between consecutive step vectors measures how much the path bends; averaging it gives a single "straightness" number (small = straight, π/2 = every step turns 90°).

**What it tells you.** Dividing the rollout's straightness by the real trajectory's puts it on a scale where 1.0 means the model reproduced the real path's geometry. Well below 1 means the model predicts a straighter, blander path than the arm actually takes — it has learned an average motion and washed out the detail.

**Reading.** ≈ 1 the rollout has realistic path geometry · ≪ 1 over-smoothed · > 1 the model adds spurious kinks.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.939 | 0.646 | 0.562 | 0.240 |

**V-JEPA (0.94) reproduces the real path geometry almost exactly.** DINO-WM smooths it somewhat (0.56–0.65). **LeWM's rollout is at 0.24** — its predicted trajectory is four times straighter than the real one; it has collapsed to a near-linear average path. This is the same failure as the high drift ratio seen from a different angle: the predictor does not track the real motion, so its open-loop rollout drifts toward a smooth default.


### Tier 3 — Action grounding

#### Action reliance  ★

**Definition.** Whether the predictor actually uses the robot's action, or whether the "action-conditioned" label is cosmetic.

**Formula.**

```
action reliance  =  ( NMSE(shuffled actions)  −  NMSE(real actions) )  /  NMSE(shuffled actions)
```

The encoder is run once; the predictor is then re-run with the real actions, and again with the actions randomly permuted across the batch (so each clip gets some other clip's action sequence — a valid action, just the wrong one).

**What it tells you.** If shuffling the action does not change the error, the predictor is ignoring it — it predicts the same "average plausible future" no matter what the robot is told to do. Such a model cannot be used for control, because control means choosing actions for their predicted effect.

**Reading.** 0 = the action is ignored · → 1 = the prediction fully depends on the action.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.015 | 0.219 | 0.163 | 0.034 |

**Only DINO-WM genuinely uses the action** (reliance 0.16 for EUPE, 0.22 for DINOv2). **V-JEPA's reliance is 0.015** — shuffling the action barely moves its error; at this checkpoint its action-conditioning is nearly inert. **LeWM's is 0.03** — also effectively unconditioned. This is a serious limitation for V-JEPA and LeWM: whatever they have learned, it is prediction of the likely future, not prediction *given a commanded action*.

#### Action effect vs. zero

**Definition.** The same test as action reliance, but comparing against zeroed rather than shuffled actions.

**Formula.**

```
effect vs. zero  =  ( NMSE(zero actions)  −  NMSE(real actions) )  /  NMSE(zero actions)
```

Zeroing the action removes the signal entirely; shuffling replaces it with a wrong signal.

**What it tells you.** Comparing the two separates "the model reacts to having *an* action" from "the model reacts to *which* action." If effect-vs-zero is much larger than reliance, the model is responding to the presence of an action input but not to its content.

**Reading.** Same scale as action reliance.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.013 | 0.127 | 0.090 | 0.022 |

For every model the two numbers are close (e.g. DINOv2: reliance 0.22, effect-vs-zero 0.13; V-JEPA: 0.015 vs 0.013), so no model is merely reacting to "an action is present" — those that respond, respond to the content.

#### Counterfactual divergence

**Definition.** How strongly the prediction moves when the action is changed, measured against how much the scene actually moves in one step.

**Formula.**

```
counterfactual divergence  =  mean_t  ‖ ẑ_t(a)  −  ẑ_t(a′) ‖  /  ‖ z_t − z_{t−1} ‖
```

a is the real action, a′ a wrong (shuffled) one. The numerator is how far apart the two predictions are; the denominator is the true one-step motion of the latent.

**What it tells you.** A model can *rely* on the action (previous metric > 0) and still be *under-responsive* — it nudges the prediction in the right direction but not far enough. ≈ 1 means the action's influence is calibrated to the real scale of motion.

**Reading.** ≈ 1 calibrated · < 1 under-responsive · > 1 over-reactive.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.061 | 0.499 | 0.439 | 0.277 |

All four are below 1, so every model that uses the action under-uses it. DINO-WM is closest (0.44–0.50 — a wrong action moves the prediction about half as much as real motion). V-JEPA is at 0.06, essentially not moving the prediction at all when the action changes, consistent with its near-zero reliance.

#### Action-inversion R²

**Definition.** Whether the *effect* of an action is recoverable from the pair of latents around it, even if the predictor does not use it.

**Formula.**

```
fit  a_t  ≈  W · [ pool(z_t) ; pool(z_{t+1}) ]  by ridge regression on a train split;
report R² on a held-out split.
```

pool(z) averages a frame's tokens into one vector. The regression asks: given the latent before and after a transition, can a linear map read off which action caused it?

**What it tells you.** This probes the *representation*, not the predictor. A high value means the action's consequences are linearly encoded in how the latent changed — the information is present. The predictor may still choose not to use it (see action reliance).

**Reading.** R² near 1 = the transition fully encodes the action · near 0 = it does not · < 0 = the linear map fails to generalise.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.401 | 0.113 | 0.101 | 0.0010 |

**V-JEPA scores 0.40** — its transitions clearly encode the action's effect, even though its predictor barely uses the action input. The information is in the representation; the predictor is leaving it on the table. DINO-WM scores ~0.11 and LeWM ~0.00: in LeWM's near-static latent, one transition looks much like another regardless of action.


### Tier 5 — Representation probing

#### Linear probe R² to end-effector state

**Definition.** How much task-relevant information (the robot's gripper position and state) is linearly readable from the latent.

**Formula.**

```
fit  s_t  ≈  W · pool(z_t)  by ridge regression, split by episode;
report the mean over the 7 state dimensions of the held-out R².
```

s_t is the 7-D BridgeData state vector (end-effector position, orientation, gripper). "Split by episode" means whole episodes are held out, so the probe cannot memorise.

**What it tells you.** A representation useful for a robot task should expose the task variables simply. This measures how much of the state a *linear* readout recovers.

**Reading.** R² near 1 = the state is linearly present · near 0 = it is not.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.525 | 1 | 1 | 1 |

**This row is not usable in the current results.** For DINO-WM and LeWM the evaluation harness fed the probe a zeroed state target (a bug: those models' adapters zero the state because their predictor does not consume it, and the probe code read that zero tensor as its target). The bug is fixed; those three models must be re-run before this metric can be reported. V-JEPA's value (R² = 0.52) is correct — its adapter passes the true state through. For V-JEPA, position dimensions probe at 0.78–0.85 and orientation/gripper at 0.13–0.48.

#### Probe transfer

**Definition.** Whether a *predicted* latent still carries the task information that a *real* latent does — i.e. does the prediction stay semantically meaningful, or just look plausible.

**Formula.**

```
probe transfer  =  R²( probe applied to ẑ )  /  R²( probe applied to z )
the probe W is fit on real latents only.
```
**What it tells you.** 1.0 means the predictor's output is as informative about the robot state as the real latent. Below 1 means the prediction has drifted somewhere that looks fine to the reconstruction loss but has lost task content.

**Reading.** ≈ 1 predictions keep the information · < 1 semantically degraded.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 1.044 | 1 | 1 | 1 |

Blocked by the same bug as the probe R² above. V-JEPA's value (1.04) is real and healthy — its one-step predictions retain the full end-effector information.


### Tier 7 — Latent-space geometry

#### Effective rank of the latent (relative to its width)

**Definition.** How many of the latent's dimensions actually carry variation — a measure of whether the representation has partially collapsed.

**Formula.**

```
effective rank  =  ( Σ_i λ_i )²  /  ( Σ_i λ_i² )     where λ_i are the eigenvalues of the
                                                    latent's covariance matrix
reported value  =  effective rank  /  D   (D = latent width: 768 for V-JEPA, 384 for the rest)
```

The covariance eigenvalues λ_i measure how much the latent varies along each principal direction. The formula (the "participation ratio") equals D if all directions vary equally and 1 if all the variance is in a single direction.

**What it tells you.** A low effective rank means the latent lives in a small subspace. Predicting a vector confined to a few dimensions is easy — so a low effective rank *inflates* the raw NMSE and R², which is another reason those are not ranked. It is diagnostic, not a score.

**Reading.** Closer to 1 = the full representation is used · near 0 = collapsed into a few directions.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.022 | 0.035 | 0.042 | 0.140 |

All four representations use a small fraction of their dimensions: V-JEPA 0.02, DINO-WM 0.035–0.042, **LeWM 0.14**. LeWM's is the *least* compressed — noteworthy because its encoder trains from scratch and could have collapsed, but its isotropy regulariser (see next metric) has kept it well spread. So LeWM's poor skill score is **not** explained by a collapsed latent; one-step prediction there is genuinely easy for a different reason (the latent barely moves).

#### Isotropy

**Definition.** Whether the latent vectors point in all directions evenly, or are squeezed into a narrow cone.

**Formula.**

```
isotropy  =  1  −  ‖ mean_i ( x_i / ‖x_i‖ ) ‖²
   x_i are the latent vectors (mean not subtracted)
```

Each x_i / ‖x_i‖ is a unit vector. If the latents point every which way, these unit vectors cancel and their mean has length near 0, so isotropy → 1. If they cluster around a common direction, the mean has length near 1 and isotropy → 0.

**What it tells you.** An anisotropic (cone-shaped) latent wastes representational capacity and tends to make similarity comparisons degenerate. This is a known issue with transformer embeddings, DINOv2's in particular.

**Reading.** 1 = perfectly spread · near 0 = collapsed to a cone.

**Results.**

| V-JEPA | DINOv2 | EUPE | LeWM |
| ---: | ---: | ---: | ---: |
| 0.032 | 0.363 | 0.252 | 0.996 |

**LeWM is almost perfectly isotropic (0.996)** — its SIGReg isotropy regulariser is doing its job. DINO-WM is moderately coned (0.25–0.36, the familiar DINOv2 anisotropy). V-JEPA is the most anisotropic at 0.03. Combined with the effective-rank numbers: V-JEPA's latent is both low-rank and coned, so its raw NMSE benefits from an easy target more than any other model — which makes its high skill-vs-persistence score all the more meaningful, since it earned it on a representation that also happens to move the most.


## Appendix A — raw latent losses (context only, never ranked)

| Quantity | V-JEPA | DINOv2 | EUPE | LeWM |
|---| ---: | ---: | ---: | ---: |
| NMSE | 0.233 | 0.113 | 0.089 | 0.005 |
| R² | 0.767 | 0.887 | 0.911 | 0.995 |
| Cosine error | 0.041 | 0.051 | 0.026 | 0.003 |
| Relative L2 | 0.268 | 0.255 | 0.201 | 0.056 |
| NMSE of the persistence baseline | 0.905 | 0.151 | 0.101 | 0.004 |
| NMSE of the constant-velocity baseline | 2.264 | 0.344 | 0.221 | 0.007 |
| NMSE of the untrained twin | 3.661 | 0.570 | 7.215 | 1.042 |

The persistence-baseline NMSE ranges from 0.004 (LeWM) to 0.90 (V-JEPA): the single clearest illustration that these representations pose prediction problems of completely different difficulty, and that a raw loss cannot be compared between them.

## Appendix B — training curves

The cheap metrics (Tiers 1–2, plus geometry) computed on every saved checkpoint (1 000 clips each). `effective rank` is expected to be exactly flat for the frozen-encoder models — a harness correctness check — and free to move for LeWM, whose encoder is trained.

**DINO-WM · DINOv2** — steps 1 000 to 16 000

- NMSE `█▄▃▂▁▁▁▁▁▁▁▁▁▁▁▁` 0.385 → 0.112
- Skill vs. persistence `▁▄▅▆▇▇▇▇▇▇▇▇▇▇█▇` -1.57 → +0.25, peak +0.25 at step 15,000
- Effective rank `▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` 13.7 → 13.7 — flat, frozen encoder ✓

**DINO-WM · EUPE** — steps 1 000 to 16 000

- NMSE `█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` 1.066 → 0.089
- Skill vs. persistence `▁▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇█` -9.74 → +0.11, peak +0.11 at step 16,000
- Effective rank `▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁` 16.0 → 16.0 — flat, frozen encoder ✓

**LeWM-bridge** — steps 1 000 to 16 000

- NMSE `█▄▄▂▂▂▁▁▁▁▁▁▁▁▁▁` 0.014 → 0.005
- Skill vs. persistence `▄▁▁▁▂▃▄▅▆▇▇▇█▇▆▅` -0.15 → -0.13, peak -0.08 at step 13,000
- Effective rank `▁▂▃▅▄▄▆▆▆▇▇█▇▇▇▇` 43.0 → 51.4 — moves, from-scratch encoder

---

## Synthesis

**On the core question — has the predictor learned action-conditioned dynamics? — V-JEPA
2.1-AC is ahead.** It is the only model with a large skill vs. persistence (0.74), the only
one whose transitions clearly encode the action (inversion R² 0.40), and the only one whose
open-loop rollout keeps the real path's geometry (straightness ratio 0.94). Its cost is a
hard target (the latent that moves the most, and the most anisotropic), and two real gaps:
its predictor barely *uses* the action (reliance 0.015), and it is only at step 16 000 of a
planned 80 000, so all of its numbers are a mid-training snapshot.

**DINO-WM is the most usable model today.** Its raw error is the lowest, but the honest
reading is that its frozen encoder produces a slowly-changing latent where persistence is
already a strong baseline; its predictor adds a real, if modest, improvement (skill 0.12–0.26).
It is the only model that genuinely responds to the commanded action (reliance 0.16–0.22,
counterfactual divergence ≈ 0.45), and its rollout is the most stable (drift ratio ≈ 2.1 at
h = 4). Its training curves are clean: skill rises steadily and plateaus around step 11 000,
and the latent geometry is constant across all 16 checkpoints — a built-in correctness check
that the harness passes. Of the two encoders, DINOv2 has the higher skill vs. persistence
(0.26 vs 0.12) and the stronger action grounding; EUPE has the lower raw error and is slightly
less converged.

**LeWM-bridge's headline error is misleading.** NMSE 0.005 and R² 0.995 look excellent, but
its skill vs. persistence is negative (−0.14): its autoregressive predictor is measurably
worse than the rule "nothing changes." This is not a collapsed representation — LeWM's latent
is the least compressed and the most isotropic of the four, because its isotropy regulariser
works. The problem is that in that representation the frame-to-frame change is tiny, so
persistence is near-optimal and the learned predictor mostly adds error, which then compounds
badly in open-loop rollout (drift ratio 6.95, straightness ratio 0.24). Its training curve
peaks around step 12 000–13 000 and *declines* to 16 000, so evaluating it at the matched step
of 16 000 slightly understates it — but its best checkpoint is still negative.

## Limitations of this report

1. **V-JEPA is 20 % trained.** Its numbers should be read as a lower bound; the comparison
   should be repeated at ≥ 40 000 steps.
2. **The Tier-5 probe rows are invalid** for three of the four models (state-target bug, now
   fixed) and require a re-run.
3. **Single seed, single evaluation set.** No confidence intervals; the evaluation set is the
   training-time held-out slice, not a fully fresh one.
4. **No shared-space results yet** (fairness level C): pixel reconstruction, a frozen judge
   encoder, and planning success are the subject of the next phase and are what would allow an
   *absolute* statement that one representation is better than another, rather than the
   relative statements above.

