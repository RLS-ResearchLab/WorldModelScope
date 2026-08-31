"""The adapter contract every world model wraps to.

Why this exists
---------------
The four models under comparison each pick their own latent space, their own
input format, their own context length, and their own "predict the next latent"
signature (V-JEPA needs a state vector; DINO-WM runs its own action encoder;
LeWM emits a single CLS vector). If every metric were re-implemented per model,
"we measured all four the same way" would be a hope, not a fact.

An adapter pushes *all* model-specific tensor bookkeeping behind a handful of
methods that return tensors in one canonical shape. Metric code then imports
nothing model-specific.

Canonical shapes
----------------
* ``frames``            : ``(B, N, H, W, 3)`` uint8 -- the canonical clip stream
                          (see ``data_spec.py``); identical bytes for every model.
* ``latents``           : ``(B, T, P, D)`` -- this model's space. ``T`` latent
                          frames, ``P`` tokens/frame (``P == 1`` for LeWM),
                          ``D`` feature dim.
* per-latent actions    : ``(B, T, A)`` -- the raw canonical actions subsampled
                          and aligned to *this model's* latent frames.

One-step alignment
------------------
``teacher_forced`` returns ``pred, target, z_prev, z_prev2`` all shaped
``(B, K, P, D)`` and mutually aligned: ``pred[:, i]`` is the model's prediction
of ``target[:, i]``; ``z_prev[:, i]`` is the latent that prediction steps *from*
(the persistence baseline); ``z_prev2[:, i]`` is one latent frame earlier (the
constant-velocity baseline ``2*z_prev - z_prev2``). ``K`` is however many
one-step predictions the model naturally makes for one clip.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass
class TeacherForced:
    """Aligned one-step teacher-forced tensors, all ``(B, K, P, D)``.

    ``target[:, i]`` is the encoder's real latent for frame ``target_start + i``;
    the runner uses ``target_start`` to line per-frame actions / states up with
    the predictions.
    """

    pred: Tensor
    target: Tensor
    z_prev: Tensor
    z_prev2: Tensor | None = None
    target_start: int = 1


@runtime_checkable
class WorldModelAdapter(Protocol):
    # ---- declared geometry (used by the runner to skip unsupported metrics) ----
    name: str
    latent_dim: int             # D
    tokens_per_frame: int       # P
    num_context_frames: int     # latent frames of context the predictor consumes
    action_dim: int             # A
    fps: float
    is_action_conditioned: bool
    has_decoder: bool
    encoder_fingerprint: str    # identical across ckpts that share a frozen encoder
    encoder_is_frozen: bool     # True -> runner encodes the eval set once and reuses
    train_step: int

    # ---- core ops ----
    def encode(self, frames: Tensor) -> Tensor:
        """``(B, N, H, W, 3)`` uint8  ->  ``(B, T, P, D)`` latents in this space."""

    def align_actions(self, actions: Tensor) -> Tensor:
        """``(B, N, A)`` canonical  ->  ``(B, T, A)`` aligned to this model's latent frames."""

    def align_states(self, states: Tensor) -> Tensor:
        """``(B, N, S)`` canonical  ->  ``(B, T, S)``. Return zeros if unused."""

    def teacher_forced(self, latents: Tensor, actions: Tensor, states: Tensor) -> TeacherForced:
        """Real context in, predict every next latent. See module docstring for alignment."""

    def rollout(self, latents: Tensor, actions: Tensor, states: Tensor, horizon: int) -> Tensor:
        """Open-loop: seed with the first ``num_context_frames`` real latents, then feed
        the model its own predictions. Returns ``(B, horizon, P, D)`` aligned so that
        output ``h`` (0-indexed) corresponds to real latent ``num_context_frames + h``."""

    def decode(self, latents: Tensor) -> Tensor:
        """Optional. ``(B, T, P, D)`` -> ``(B, T, 3, H, W)`` in [0, 1]. Raises if no decoder."""

    def build_untrained(self) -> "WorldModelAdapter":
        """Same architecture and same (frozen) encoder, predictor re-initialised at random.
        The Tier-1 'skill vs untrained twin' control: what did *training* add over a random
        predictor on the identical frozen latent space."""


class BaseAdapter:
    """Optional base class providing the generic autoregressive ``rollout`` in terms of a
    concrete ``_predict_next`` step. Concrete adapters may still override ``rollout`` to
    delegate to a model's own (e.g. V-JEPA / DINO-WM ship one)."""

    num_context_frames: int
    tokens_per_frame: int

    def _predict_next(self, ctx_latents: Tensor, ctx_actions: Tensor, ctx_states: Tensor) -> Tensor:
        """``(B, C, P, D)`` context (+ aligned actions/states) -> ``(B, P, D)`` next latent."""
        raise NotImplementedError

    def rollout(self, latents: Tensor, actions: Tensor, states: Tensor, horizon: int) -> Tensor:
        C = self.num_context_frames
        ctx = latents[:, :C].clone()                       # (B, C, P, D)
        preds = []
        for h in range(horizon):
            a = actions[:, h:h + C]
            s = states[:, h:h + C]
            nxt = self._predict_next(ctx, a, s)            # (B, P, D)
            preds.append(nxt)
            ctx = torch.cat([ctx[:, 1:], nxt[:, None]], dim=1)
        return torch.stack(preds, dim=1)                   # (B, horizon, P, D)


def sliding_windows(latents: Tensor, context: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Helper for ``teacher_forced``: from ``(B, T, P, D)`` build every length-``context``
    window and its next-frame target. Returns ``(ctx, target, z_prev, z_prev2)`` where
    ``ctx`` is ``(B, K, context, P, D)`` and the rest are ``(B, K, P, D)``.
    ``K = T - context``."""
    B, T, P, D = latents.shape
    K = T - context
    if K <= 0:
        raise ValueError(f"clip has {T} latent frames, need > context={context}")
    ctx = torch.stack([latents[:, i:i + context] for i in range(K)], dim=1)
    target = latents[:, context:context + K]
    z_prev = latents[:, context - 1:context - 1 + K]
    z_prev2 = latents[:, context - 2:context - 2 + K] if context >= 2 else None
    return ctx, target, z_prev, z_prev2
