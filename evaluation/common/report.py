"""Generate the jury-facing evaluation report from results/<model>/*.json.

    .venv/bin/python -m evaluation.common.report        -> results/REPORT.md

Every metric is written out with its definition, its formula, what it tells a
reader who has never seen it, how to read the number, and the value obtained for
each of the four models with an interpretation. Numbers in the tables are read
live from the result JSON; the surrounding prose is fixed to the step-16k
checkpoints and should be revised if those change.
"""
from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_RESULTS = _REPO / "results"
_ORDER = ["vjepa2_ac", "dino_wm_dinov2", "dino_wm_eupe", "lewm_bridge"]
_SHORT = {"vjepa2_ac": "V-JEPA 2.1-AC", "dino_wm_dinov2": "DINO-WM · DINOv2",
          "dino_wm_eupe": "DINO-WM · EUPE", "lewm_bridge": "LeWM-bridge"}
_TH = {"vjepa2_ac": "V-JEPA", "dino_wm_dinov2": "DINOv2", "dino_wm_eupe": "EUPE",
       "lewm_bridge": "LeWM"}


# ======================================================================= intro
_INTRO = r"""# Evaluating Four World Models on BridgeData V2

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
"""


# ==================================================================== metrics
# Each entry: (tier_title, name, definition, formula_lines, glossary, tells, reading, interpretation)
def _M(tier, name, definition, formula, glossary, tells, reading, interp):
    return dict(tier=tier, name=name, definition=definition, formula=formula,
                glossary=glossary, tells=tells, reading=reading, interp=interp,
                keys=[])  # keys filled per-metric below


TIER1 = "Tier 1 — One-step prediction accuracy"
TIER2 = "Tier 2 — Open-loop rollout stability"
TIER3 = "Tier 3 — Action grounding"
TIER5 = "Tier 5 — Representation probing"
TIER7 = "Tier 7 — Latent-space geometry"

METRICS: list[dict] = [
 # ---------------- Tier 1 ----------------
 {**_M(TIER1, "Normalised mean-squared error (NMSE)",
   "Teacher-forced: the predictor is given the real past latents and predicts the next one; "
   "NMSE is the squared prediction error divided by the natural spread of the target.",
   ["NMSE  =  ( Σ_t ‖ ẑ_t − z_t ‖² )  /  ( Σ_t ‖ z_t − z̄ ‖² )"],
   "The numerator is the model's total squared error. The denominator is the error of the "
   "*best constant predictor* — one that ignores the input and always outputs the average "
   "latent z̄ — so it equals the total variance of the target.",
   "It answers: is the predictor doing anything at all? NMSE = 1 means the model is no better "
   "than always guessing the average; NMSE = 0 is perfect. It is **not comparable between "
   "models** because each z lives in a different space with a different natural spread.",
   "0 perfect · 1 = as good as guessing the mean · > 1 = worse than guessing the mean.",
   "The four NMSEs span two orders of magnitude — 0.005 for LeWM to 0.23 for V-JEPA — and this "
   "spread is almost entirely an artefact of how much each representation moves between frames, "
   "not of predictor quality. This is precisely why the report does not rank on it. The "
   "numbers are shown so the reader can see the problem the skill scores solve."),
  "block": "one_step", "key": "nmse", "dir": "none"},

 {**_M(TIER1, "Coefficient of determination (R²)",
   "A rescaling of NMSE onto the usual \"fraction of variance explained\" axis.",
   ["R²  =  1 − NMSE"],
   "",
   "R² = 1 is a perfect predictor, R² = 0 is the constant-mean predictor, R² < 0 is worse than "
   "that. Same caveat as NMSE: not comparable across models.",
   "1 perfect · 0 mean-predictor · < 0 worse than the mean.",
   "All four sit above 0.76, and LeWM reaches 0.995 — but, again, on targets of wildly "
   "different difficulty. Read it alongside the skill scores below, never alone."),
  "block": "one_step", "key": "r2", "dir": "none"},

 {**_M(TIER1, "Cosine error",
   "One minus the average cosine similarity between the predicted and true next latent — a "
   "measure of *direction* only, with the magnitude divided out.",
   ["cosine error  =  1  −  mean_t  ⟨ ẑ_t , z_t ⟩ / ( ‖ ẑ_t ‖ · ‖ z_t ‖ )"],
   "⟨ẑ, z⟩ / (‖ẑ‖‖z‖) is the cosine of the angle between the two vectors: 1 if they point the "
   "same way, 0 if orthogonal, −1 if opposite.",
   "It separates two kinds of error. If the cosine error is small but NMSE is still large, the "
   "predictor points in the right direction but gets the *length* wrong (a scale error). If the "
   "cosine error is large, the predictor is pointing the wrong way entirely.",
   "0 = perfect direction · larger = worse direction. Unlike NMSE it is roughly comparable "
   "across models because it is scale-free by construction.",
   "Every model has a small cosine error (0.003–0.05), so all four get the *direction* of the "
   "next latent almost right. For V-JEPA the cosine error is 0.04 while its NMSE is 0.23 — the "
   "gap is a scale error, not a wrong prediction. The DINO-WM · DINOv2 cosine error (0.05) is "
   "the largest, consistent with its being the least converged of the DINO pair."),
  "block": "one_step", "key": "cosine_error", "dir": "low"},

 {**_M(TIER1, "Relative L2 error",
   "The prediction error as a fraction of the target's own length, averaged per sample.",
   ["relative L2  =  mean_t  ‖ ẑ_t − z_t ‖ / ‖ z_t ‖"],
   "",
   "A robustness cross-check on NMSE: NMSE squares the error and normalises by a single global "
   "variance, so a few large targets can dominate it; relative L2 normalises each sample by its "
   "own norm and does not square, so it is less sensitive to outliers.",
   "0 = perfect · smaller is better. Not comparable across models (still in latent units).",
   "The relative L2 ordering matches NMSE (LeWM lowest at 0.06, V-JEPA highest at 0.27), "
   "confirming that no single outlier clip is driving the NMSE picture."),
  "block": "one_step", "key": "relative_l2", "dir": "none"},

 {**_M(TIER1, "Skill vs. persistence  ★",
   "How much the predictor beats the trivial rule \"the next latent equals the current "
   "latent.\" This is the single most important number for whether the model learned dynamics.",
   ["skill  =  1  −  NMSE(model)  /  NMSE(persistence)",
    "NMSE(persistence)  =  ( Σ_t ‖ z_{t−1} − z_t ‖² )  /  ( Σ_t ‖ z_t − z̄ ‖² )"],
   "The *persistence* predictor outputs the previous latent unchanged — it assumes the world "
   "does not move. Both NMSEs are computed in the same latent space, so their ratio is "
   "dimensionless and **comparable across models**.",
   "Beating persistence requires predicting *change*: motion of the arm, of objects, of the "
   "scene. A model that has only learned \"the next frame looks like this frame\" scores 0. "
   "Only a model that anticipates motion scores above 0.",
   "> 0 the predictor beats doing nothing · ≈ 0 it is essentially an identity map · < 0 it is "
   "*worse* than doing nothing.",
   "This is where the models separate sharply. **V-JEPA scores 0.74** — its predictor's error "
   "is only a quarter of persistence's, so it has genuinely learned to predict how the scene "
   "moves. **DINO-WM adds 0.12–0.26**: a real improvement, but persistence is already a strong "
   "baseline in its slow-moving representation, so there is little headroom. **LeWM scores "
   "−0.14** — its predictor is measurably worse than simply repeating the last latent. In "
   "LeWM's representation the frame-to-frame change is so small (see NMSE of persistence = "
   "0.004 in the appendix) that persistence is near-optimal and the learned predictor only "
   "adds noise."),
  "block": "one_step", "key": "skill_vs_persistence", "dir": "high"},

 {**_M(TIER1, "Skill vs. constant-velocity",
   "How much the predictor beats linear extrapolation of the last two latents.",
   ["skill  =  1  −  NMSE(model)  /  NMSE(constant-velocity)",
    "constant-velocity prediction:  ẑ_t  =  2 z_{t−1}  −  z_{t−2}"],
   "The constant-velocity baseline assumes the latent keeps moving in a straight line at "
   "constant speed — the next-simplest model of motion after persistence.",
   "Beating it means the predictor has learned something about *how* motion changes — "
   "acceleration, direction changes, contact events — not just that motion continues.",
   "> 0 the model learned beyond straight-line motion.",
   "All four beat constant velocity (0.25–0.90), so every predictor has learned at least some "
   "non-linear structure. V-JEPA's 0.90 is the strongest; for it, linear extrapolation in the "
   "latent is actually *worse than the mean* (NMSE 2.26, appendix), meaning its latent "
   "trajectory is strongly curved."),
  "block": "one_step", "key": "skill_vs_const_velocity", "dir": "high"},

 {**_M(TIER1, "Skill vs. untrained twin  ★",
   "How much *training the predictor* helped, with the effect of an easy or hard "
   "representation removed.",
   ["skill  =  1  −  NMSE(model)  /  NMSE(untrained twin)"],
   "The *untrained twin* is the exact same architecture — same frozen encoder, same predictor "
   "network — but with the predictor's weights left at their random initialisation. It is "
   "evaluated on the identical clips.",
   "Persistence and constant-velocity are hand-written baselines; the untrained twin is the "
   "*model itself before learning*. A high score means training moved the predictor far from "
   "random. Because the encoder is identical, this isolates the contribution of predictor "
   "training from the difficulty of the latent space.",
   "≈ 1 training did almost all the work · ≈ 0 a random predictor on this encoder does as well.",
   "All four score ≥ 0.80, so in every case predictor training clearly mattered relative to "
   "random weights. Note that this can be high even when skill vs. persistence is low or "
   "negative (LeWM: 0.995 here, −0.14 there): training moved LeWM's predictor a long way from "
   "random, it just moved it to a place that is still worse than the persistence rule."),
  "block": "one_step", "key": "skill_vs_untrained", "dir": "high"},

 # ---------------- Tier 2 ----------------
 {**_M(TIER2, "Rollout NMSE at horizon h, and the drift ratio  ★",
   "When the model is run open-loop — fed its own predictions instead of the real frames — "
   "errors accumulate. The drift ratio measures how fast.",
   ["NMSE(h)  =  the Tier-1 NMSE, but ẑ_{t+h} comes from h autoregressive steps",
    "drift ratio(h)  =  NMSE(h)  /  NMSE(1)"],
   "NMSE(1) is one step from real context (close to the teacher-forced NMSE). NMSE(h) for h > 1 "
   "feeds the model's own output back in h − 1 times.",
   "Dividing by NMSE(1) removes \"how good is a single step\" and leaves only the *compounding* "
   "of error. Because it is a ratio of two NMSEs in the same space, the drift ratio is "
   "**directly comparable across models** — it is the cleanest rollout number. A flat drift "
   "ratio means a stable model you can plan with; a steep one means the model diverges.",
   "1.0 = error does not grow with horizon · larger = the model's own mistakes feed back and "
   "amplify. Reported here at h = 4.",
   "V-JEPA (2.18), DINOv2 (2.13) and EUPE (2.22) are close: after four open-loop steps their "
   "error is roughly doubled — moderate, controlled drift. **LeWM's drift ratio is 6.95** — "
   "after four steps its error is seven times the one-step error. Its rollout is unusable more "
   "than two or three steps out, which matters for any downstream planning use."),
  "block": "rollout", "key": "_drift4", "dir": "low"},

 {**_M(TIER2, "Usable horizon",
   "The furthest the model can be rolled forward before it explains less than half the "
   "variance of the true trajectory.",
   ["usable horizon  =  max { h  :  R²(h) ≥ 0.5 }",
    "R²(h)  =  1 − NMSE(h)"],
   "",
   "A single integer summarising the rollout curve. It is capped by the clip length — 7 "
   "predicted steps for V-JEPA (its encoder produces 8 latent frames from 16 images), 13 for "
   "DINO-WM and LeWM — so it should be read together with the drift ratio, which is not capped.",
   "Higher is better; the maximum is the number of predictable frames in the clip.",
   "DINO-WM and LeWM both reach the cap of 13: in *absolute* terms their rollout NMSE never "
   "crosses 0.5 within the clip. V-JEPA's is 3. But note the drift ratio tells a different "
   "story for LeWM — its *absolute* rollout error stays low only because its latent barely "
   "moves; *relative* to its own one-step error it diverges fastest of the four."),
  "block": "rollout", "key": "usable_horizon", "dir": "high"},

 {**_M(TIER2, "Compounding rate",
   "The exponential rate at which rollout error grows per step.",
   ["fit  log NMSE(h)  ≈  α  +  β·h   by least squares;   compounding rate  =  β"],
   "If error grew purely exponentially, NMSE(h) = NMSE(1)·e^{β(h−1)} and β would be its growth "
   "constant.",
   "A second, slope-based view of the same rollout curve as the drift ratio. Larger β means "
   "faster blow-up.",
   "Smaller is better; 0 would mean no growth.",
   "V-JEPA 0.14, DINO-WM ~0.10, LeWM 0.22 — the same ordering as the drift ratio, with LeWM "
   "compounding fastest."),
  "block": "rollout", "key": "compounding_rate", "dir": "low"},

 {**_M(TIER2, "Rollout path straightness, relative to the real path",
   "Whether the model's rolled-out trajectory has the same shape as the true trajectory, or a "
   "smoothed / distorted one.",
   ["straightness(sequence)  =  mean_t  arccos( ⟨ Δz_t , Δz_{t+1} ⟩ / ( ‖Δz_t‖ · ‖Δz_{t+1}‖ ) )",
    "   where  Δz_t  =  z_{t+1} − z_t     (the step vector at time t)",
    "reported value  =  straightness(rollout)  /  straightness(real trajectory)"],
   "Δz_t is the direction the latent moves from frame t to t+1. The turning angle between "
   "consecutive step vectors measures how much the path bends; averaging it gives a single "
   "\"straightness\" number (small = straight, π/2 = every step turns 90°).",
   "Dividing the rollout's straightness by the real trajectory's puts it on a scale where 1.0 "
   "means the model reproduced the real path's geometry. Well below 1 means the model predicts "
   "a straighter, blander path than the arm actually takes — it has learned an average motion "
   "and washed out the detail.",
   "≈ 1 the rollout has realistic path geometry · ≪ 1 over-smoothed · > 1 the model adds "
   "spurious kinks.",
   "**V-JEPA (0.94) reproduces the real path geometry almost exactly.** DINO-WM smooths it "
   "somewhat (0.56–0.65). **LeWM's rollout is at 0.24** — its predicted trajectory is four "
   "times straighter than the real one; it has collapsed to a near-linear average path. This "
   "is the same failure as the high drift ratio seen from a different angle: the predictor "
   "does not track the real motion, so its open-loop rollout drifts toward a smooth default."),
  "block": "rollout", "key": "_straight_ratio", "dir": "one"},

 # ---------------- Tier 3 ----------------
 {**_M(TIER3, "Action reliance  ★",
   "Whether the predictor actually uses the robot's action, or whether the "
   "\"action-conditioned\" label is cosmetic.",
   ["action reliance  =  ( NMSE(shuffled actions)  −  NMSE(real actions) )  /  NMSE(shuffled actions)"],
   "The encoder is run once; the predictor is then re-run with the real actions, and again "
   "with the actions randomly permuted across the batch (so each clip gets some other clip's "
   "action sequence — a valid action, just the wrong one).",
   "If shuffling the action does not change the error, the predictor is ignoring it — it "
   "predicts the same \"average plausible future\" no matter what the robot is told to do. "
   "Such a model cannot be used for control, because control means choosing actions for their "
   "predicted effect.",
   "0 = the action is ignored · → 1 = the prediction fully depends on the action.",
   "**Only DINO-WM genuinely uses the action** (reliance 0.16 for EUPE, 0.22 for DINOv2). "
   "**V-JEPA's reliance is 0.015** — shuffling the action barely moves its error; at this "
   "checkpoint its action-conditioning is nearly inert. **LeWM's is 0.03** — also effectively "
   "unconditioned. This is a serious limitation for V-JEPA and LeWM: whatever they have "
   "learned, it is prediction of the likely future, not prediction *given a commanded action*."),
  "block": "action", "key": "action_reliance", "dir": "high"},

 {**_M(TIER3, "Action effect vs. zero",
   "The same test as action reliance, but comparing against zeroed rather than shuffled "
   "actions.",
   ["effect vs. zero  =  ( NMSE(zero actions)  −  NMSE(real actions) )  /  NMSE(zero actions)"],
   "Zeroing the action removes the signal entirely; shuffling replaces it with a wrong signal.",
   "Comparing the two separates \"the model reacts to having *an* action\" from \"the model "
   "reacts to *which* action.\" If effect-vs-zero is much larger than reliance, the model is "
   "responding to the presence of an action input but not to its content.",
   "Same scale as action reliance.",
   "For every model the two numbers are close (e.g. DINOv2: reliance 0.22, effect-vs-zero "
   "0.13; V-JEPA: 0.015 vs 0.013), so no model is merely reacting to \"an action is present\" "
   "— those that respond, respond to the content."),
  "block": "action", "key": "action_effect_vs_zero", "dir": "high"},

 {**_M(TIER3, "Counterfactual divergence",
   "How strongly the prediction moves when the action is changed, measured against how much "
   "the scene actually moves in one step.",
   ["counterfactual divergence  =  mean_t  ‖ ẑ_t(a)  −  ẑ_t(a′) ‖  /  ‖ z_t − z_{t−1} ‖"],
   "a is the real action, a′ a wrong (shuffled) one. The numerator is how far apart the two "
   "predictions are; the denominator is the true one-step motion of the latent.",
   "A model can *rely* on the action (previous metric > 0) and still be *under-responsive* — "
   "it nudges the prediction in the right direction but not far enough. ≈ 1 means the action's "
   "influence is calibrated to the real scale of motion.",
   "≈ 1 calibrated · < 1 under-responsive · > 1 over-reactive.",
   "All four are below 1, so every model that uses the action under-uses it. DINO-WM is "
   "closest (0.44–0.50 — a wrong action moves the prediction about half as much as real "
   "motion). V-JEPA is at 0.06, essentially not moving the prediction at all when the action "
   "changes, consistent with its near-zero reliance."),
  "block": "action", "key": "counterfactual_divergence", "dir": "one"},

 {**_M(TIER3, "Action-inversion R²",
   "Whether the *effect* of an action is recoverable from the pair of latents around it, even "
   "if the predictor does not use it.",
   ["fit  a_t  ≈  W · [ pool(z_t) ; pool(z_{t+1}) ]  by ridge regression on a train split;",
    "report R² on a held-out split."],
   "pool(z) averages a frame's tokens into one vector. The regression asks: given the latent "
   "before and after a transition, can a linear map read off which action caused it?",
   "This probes the *representation*, not the predictor. A high value means the action's "
   "consequences are linearly encoded in how the latent changed — the information is present. "
   "The predictor may still choose not to use it (see action reliance).",
   "R² near 1 = the transition fully encodes the action · near 0 = it does not · < 0 = the "
   "linear map fails to generalise.",
   "**V-JEPA scores 0.40** — its transitions clearly encode the action's effect, even though "
   "its predictor barely uses the action input. The information is in the representation; the "
   "predictor is leaving it on the table. DINO-WM scores ~0.11 and LeWM ~0.00: in LeWM's "
   "near-static latent, one transition looks much like another regardless of action."),
  "block": "action", "key": "action_inversion_r2", "dir": "high"},

 # ---------------- Tier 5 ----------------
 {**_M(TIER5, "Linear probe R² to end-effector state",
   "How much task-relevant information (the robot's gripper position and state) is linearly "
   "readable from the latent.",
   ["fit  s_t  ≈  W · pool(z_t)  by ridge regression, split by episode;",
    "report the mean over the 7 state dimensions of the held-out R²."],
   "s_t is the 7-D BridgeData state vector (end-effector position, orientation, gripper). "
   "\"Split by episode\" means whole episodes are held out, so the probe cannot memorise.",
   "A representation useful for a robot task should expose the task variables simply. This "
   "measures how much of the state a *linear* readout recovers.",
   "R² near 1 = the state is linearly present · near 0 = it is not.",
   "**This row is not usable in the current results.** For DINO-WM and LeWM the evaluation "
   "harness fed the probe a zeroed state target (a bug: those models' adapters zero the state "
   "because their predictor does not consume it, and the probe code read that zero tensor as "
   "its target). The bug is fixed; those three models must be re-run before this metric can be "
   "reported. V-JEPA's value (R² = 0.52) is correct — its adapter passes the true state "
   "through. For V-JEPA, position dimensions probe at 0.78–0.85 and orientation/gripper at "
   "0.13–0.48."),
  "block": "probe", "key": "probe_r2_real", "dir": "high"},

 {**_M(TIER5, "Probe transfer",
   "Whether a *predicted* latent still carries the task information that a *real* latent does — "
   "i.e. does the prediction stay semantically meaningful, or just look plausible.",
   ["probe transfer  =  R²( probe applied to ẑ )  /  R²( probe applied to z )",
    "the probe W is fit on real latents only."],
   "",
   "1.0 means the predictor's output is as informative about the robot state as the real "
   "latent. Below 1 means the prediction has drifted somewhere that looks fine to the "
   "reconstruction loss but has lost task content.",
   "≈ 1 predictions keep the information · < 1 semantically degraded.",
   "Blocked by the same bug as the probe R² above. V-JEPA's value (1.04) is real and healthy — "
   "its one-step predictions retain the full end-effector information."),
  "block": "probe", "key": "probe_transfer", "dir": "one"},

 # ---------------- Tier 7 ----------------
 {**_M(TIER7, "Effective rank of the latent (relative to its width)",
   "How many of the latent's dimensions actually carry variation — a measure of whether the "
   "representation has partially collapsed.",
   ["effective rank  =  ( Σ_i λ_i )²  /  ( Σ_i λ_i² )     where λ_i are the eigenvalues of the",
    "                                                    latent's covariance matrix",
    "reported value  =  effective rank  /  D   (D = latent width: 768 for V-JEPA, 384 for the rest)"],
   "The covariance eigenvalues λ_i measure how much the latent varies along each principal "
   "direction. The formula (the \"participation ratio\") equals D if all directions vary "
   "equally and 1 if all the variance is in a single direction.",
   "A low effective rank means the latent lives in a small subspace. Predicting a vector "
   "confined to a few dimensions is easy — so a low effective rank *inflates* the raw NMSE and "
   "R², which is another reason those are not ranked. It is diagnostic, not a score.",
   "Closer to 1 = the full representation is used · near 0 = collapsed into a few directions.",
   "All four representations use a small fraction of their dimensions: V-JEPA 0.02, DINO-WM "
   "0.035–0.042, **LeWM 0.14**. LeWM's is the *least* compressed — noteworthy because its "
   "encoder trains from scratch and could have collapsed, but its isotropy regulariser (see "
   "next metric) has kept it well spread. So LeWM's poor skill score is **not** explained by a "
   "collapsed latent; one-step prediction there is genuinely easy for a different reason (the "
   "latent barely moves)."),
  "block": "geometry", "key": "effective_rank_ratio", "dir": "none"},

 {**_M(TIER7, "Isotropy",
   "Whether the latent vectors point in all directions evenly, or are squeezed into a narrow "
   "cone.",
   ["isotropy  =  1  −  ‖ mean_i ( x_i / ‖x_i‖ ) ‖²",
    "   x_i are the latent vectors (mean not subtracted)"],
   "Each x_i / ‖x_i‖ is a unit vector. If the latents point every which way, these unit "
   "vectors cancel and their mean has length near 0, so isotropy → 1. If they cluster around a "
   "common direction, the mean has length near 1 and isotropy → 0.",
   "An anisotropic (cone-shaped) latent wastes representational capacity and tends to make "
   "similarity comparisons degenerate. This is a known issue with transformer embeddings, "
   "DINOv2's in particular.",
   "1 = perfectly spread · near 0 = collapsed to a cone.",
   "**LeWM is almost perfectly isotropic (0.996)** — its SIGReg isotropy regulariser is doing "
   "its job. DINO-WM is moderately coned (0.25–0.36, the familiar DINOv2 anisotropy). V-JEPA "
   "is the most anisotropic at 0.03. Combined with the effective-rank numbers: V-JEPA's latent "
   "is both low-rank and coned, so its raw NMSE benefits from an easy target more than any "
   "other model — which makes its high skill-vs-persistence score all the more meaningful, "
   "since it earned it on a representation that also happens to move the most."),
  "block": "geometry", "key": "isotropy", "dir": "none"},
]


# ================================================================== assembly
_APPENDIX_KEYS = [
    ("nmse", "NMSE"), ("r2", "R²"), ("cosine_error", "Cosine error"),
    ("relative_l2", "Relative L2"), ("nmse_persistence", "NMSE of the persistence baseline"),
    ("nmse_const_velocity", "NMSE of the constant-velocity baseline"),
    ("nmse_untrained_twin", "NMSE of the untrained twin"),
]

_CONCLUSION = r"""---

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
"""


def _load():
    full, curves = {}, {}
    for m in _ORDER:
        f = _RESULTS / m / "step_016000.json"
        if f.exists():
            full[m] = json.loads(f.read_text())
        c = _RESULTS / m / "curve.json"
        if c.exists():
            curves[m] = json.loads(c.read_text())
    return full, curves


def _value(d: dict, block: str, key: str):
    b = d.get(block, {})
    if key == "_drift4":
        dr = b.get("drift_ratio_by_h", {})
        return dr.get("4") or dr.get(4)
    if key == "_straight_ratio":
        sp, sr = b.get("straightness_pred"), b.get("straightness_real")
        return sp / sr if sp and sr else None
    return b.get(key)


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) or (isinstance(v, float) and float(v).is_integer()):
        return str(int(v))
    if abs(v) < 0.001:
        return f"{v:.4f}"
    return f"{v:.3f}"


def _table(full, models, block, key):
    L = ["| " + " | ".join(_TH[m] for m in models) + " |",
         "|" + "|".join([" ---: "] * len(models)) + "|",
         "| " + " | ".join(_fmt(_value(full[m], block, key)) for m in models) + " |"]
    return "\n".join(L)


def _sparkline(xs):
    bars = "▁▂▃▄▅▆▇█"
    xs = [x for x in xs if isinstance(x, (int, float))]
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-12:
        return bars[0] * len(xs)
    return "".join(bars[min(7, int((x - lo) / (hi - lo) * 7))] for x in xs)


def build() -> str:
    full, curves = _load()
    models = [m for m in _ORDER if m in full]
    P: list[str] = [_INTRO]

    # headline
    P.append("## Headline numbers\n")
    P.append("All four models, evaluated at training step 16 000 on the 4 106-clip held-out set. "
             "★ marks the metrics the comparison is ranked on.\n")
    head = [("one_step", "skill_vs_persistence", "Skill vs. persistence ★", "higher"),
            ("one_step", "skill_vs_untrained", "Skill vs. untrained twin ★", "higher"),
            ("rollout", "_drift4", "Drift ratio at h = 4 ★", "lower"),
            ("rollout", "usable_horizon", "Usable horizon", "higher"),
            ("rollout", "_straight_ratio", "Rollout straightness / real ★", "≈ 1"),
            ("action", "action_reliance", "Action reliance ★", "higher"),
            ("action", "action_inversion_r2", "Action-inversion R²", "higher")]
    P.append("| Metric | " + " | ".join(_TH[m] for m in models) + " | Better |")
    P.append("|---|" + "|".join([" ---: "] * len(models)) + "|:--:|")
    for blk, key, lbl, better in head:
        P.append(f"| {lbl} | " + " | ".join(_fmt(_value(full[m], blk, key)) for m in models)
                 + f" | {better} |")
    P.append("")

    # per-metric
    P.append("## The metrics in full\n")
    P.append("For each metric: its definition, its formula with every symbol spelled out, what it "
             "tells a reader, how to read the number, and the result for each model with an "
             "interpretation.\n")
    cur = None
    for m in METRICS:
        if m["tier"] != cur:
            cur = m["tier"]
            P.append(f"\n### {cur}\n")
        P.append(f"#### {m['name']}\n")
        P.append(f"**Definition.** {m['definition']}\n")
        P.append("**Formula.**\n")
        P.append("```")
        P.extend(m["formula"])
        P.append("```")
        if m["glossary"]:
            P.append(f"\n{m['glossary']}\n")
        P.append(f"**What it tells you.** {m['tells']}\n")
        P.append(f"**Reading.** {m['reading']}\n")
        P.append("**Results.**\n")
        P.append(_table(full, models, m["block"], m["key"]))
        P.append("")
        P.append(m["interp"])
        P.append("")

    # appendix raw
    P.append("\n## Appendix A — raw latent losses (context only, never ranked)\n")
    P.append("| Quantity | " + " | ".join(_TH[m] for m in models) + " |")
    P.append("|---|" + "|".join([" ---: "] * len(models)) + "|")
    for k, lbl in _APPENDIX_KEYS:
        P.append(f"| {lbl} | " + " | ".join(_fmt(_value(full[m], 'one_step', k)) for m in models) + " |")
    P.append("\nThe persistence-baseline NMSE ranges from 0.004 (LeWM) to 0.90 (V-JEPA): the "
             "single clearest illustration that these representations pose prediction problems of "
             "completely different difficulty, and that a raw loss cannot be compared between them.\n")

    # appendix curves
    if curves:
        P.append("## Appendix B — training curves\n")
        P.append("The cheap metrics (Tiers 1–2, plus geometry) computed on every saved checkpoint "
                 "(1 000 clips each). `effective rank` is expected to be exactly flat for the "
                 "frozen-encoder models — a harness correctness check — and free to move for LeWM, "
                 "whose encoder is trained.\n")
        for m in models:
            if m not in curves:
                continue
            st = curves[m]["steps"]
            xs = [s["step"] for s in st]
            nm = [s["one_step"]["nmse"] for s in st]
            sk = [s["one_step"].get("skill_vs_persistence") for s in st]
            er = [s["geometry"]["effective_rank"] for s in st]
            sk_ok = [x for x in sk if isinstance(x, (int, float))]
            bi = max(range(len(sk)), key=lambda i: sk[i] if isinstance(sk[i], (int, float)) else -9)
            flat = max(er) - min(er) < 0.5
            P.append(f"**{_SHORT[m]}** — steps {xs[0]//1000} 000 to {xs[-1]//1000} 000\n")
            P.append(f"- NMSE `{_sparkline(nm)}` {nm[0]:.3f} → {nm[-1]:.3f}")
            if sk_ok:
                P.append(f"- Skill vs. persistence `{_sparkline(sk_ok)}` {sk_ok[0]:+.2f} → "
                         f"{sk_ok[-1]:+.2f}, peak {max(sk_ok):+.2f} at step {xs[bi]:,}")
            P.append(f"- Effective rank `{_sparkline(er)}` {er[0]:.1f} → {er[-1]:.1f} — "
                     f"{'flat, frozen encoder ✓' if flat else 'moves, from-scratch encoder'}\n")

    P.append(_CONCLUSION)
    return "\n".join(P)


def main() -> None:
    md = build()
    out = _RESULTS / "REPORT.md"
    out.write_text(md + "\n")
    print(f"wrote {out}  ({len(md.splitlines())} lines, {len(md) // 1000} KB)")


if __name__ == "__main__":
    main()
