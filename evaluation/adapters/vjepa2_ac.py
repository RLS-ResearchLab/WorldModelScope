"""V-JEPA 2.1-AC -> WorldModelAdapter.

Thin wrapper over the existing :class:`evaluation.vjepa_ac.model.VJEPA2AC`
(frozen V-JEPA 2.1 ViT-B encoder + trained action-conditioned predictor).

Geometry: 16 canonical frames -> tubelet-2 -> 8 latent frames x 196 tokens x 768.
The predictor needs a per-latent-frame 7-D state vector as well as the action.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import Tensor

from evaluation.common.adapter import TeacherForced
from evaluation.vjepa_ac.model import LATENT_FRAMES, VJEPA2AC

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_ENCODER = _REPO / "checkpoints/vjepa2_1_base/vjepa2_1_vitb_dist_vitG_384.pt"

# kwargs of VJEPA2AC's predictor -- replicated so build_untrained() can make a
# random-weight twin without touching the trained checkpoint.
_PRED_KWARGS = dict(
    img_size=224, patch_size=16, num_frames=LATENT_FRAMES, tubelet_size=1,
    embed_dim=768, predictor_embed_dim=768, depth=12, num_heads=12,
    use_rope=False, use_activation_checkpointing=False,
)


class VJEPA2ACAdapter:
    name = "vjepa2_ac"
    latent_dim = 768
    tokens_per_frame = 196
    num_context_frames = 1          # rollout seeds with 1 real latent frame, then grows
    action_dim = 7
    fps = 4.0
    is_action_conditioned = True
    has_decoder = False
    encoder_is_frozen = True

    def __init__(
        self,
        predictor_ckpt: str,
        encoder_ckpt: str | None = None,
        device: str = "cuda",
        _predictor: torch.nn.Module | None = None,
    ):
        encoder_ckpt = encoder_ckpt or str(_DEFAULT_ENCODER)
        self.device = device
        self.model = VJEPA2AC(encoder_ckpt, predictor_ckpt, device=device)
        self._predictor = _predictor or self.model.predictor
        self.train_step = self.model.train_step
        self.encoder_fingerprint = f"vjepa2_1_base::{Path(encoder_ckpt).name}"
        self.param_counts = self.model.param_counts

    # ---- encode / align ----
    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """(B, 16, 224, 224, 3) uint8 -> (B, 8, 196, 768)."""
        video = self.model.preprocess_frames(frames)
        z = self.model.encode(video)                       # (B, 8*196, 768)
        B = z.shape[0]
        return z.view(B, LATENT_FRAMES, self.tokens_per_frame, self.latent_dim)

    def align_actions(self, actions: Tensor) -> Tensor:
        """(B, 16, 7) -> (B, 8, 7): the action at each odd canonical frame, one per
        latent frame (matches the training script's `a[idx[1::2]]`)."""
        return actions[:, 1::2].to(self.device)

    def align_states(self, states: Tensor) -> Tensor:
        """(B, 16, 7) -> (B, 8, 7): the state at each even canonical frame."""
        return states[:, ::2].to(self.device)

    # ---- predictor calls ----
    @torch.no_grad()
    def _predict(self, latents_bt: Tensor, actions: Tensor, states: Tensor) -> Tensor:
        """(B, t*P, D) + (B, t, A/S) -> (B, t*P, D)."""
        with torch.autocast(self.device, dtype=torch.bfloat16):
            return self._predictor(
                latents_bt.to(self.device), actions.to(self.device), states.to(self.device)
            ).float()

    @torch.no_grad()
    def teacher_forced(self, latents: Tensor, actions: Tensor, states: Tensor) -> TeacherForced:
        B, T, P, D = latents.shape                         # T = 8
        y = self._predict(latents.reshape(B, T * P, D), actions, states)
        pred = y[:, :-P].reshape(B, T - 1, P, D)           # predictions of frames 1..T-1
        target = latents[:, 1:]
        z_prev = latents[:, :-1]                           # frame each prediction steps from
        # const-velocity needs frame i-1; undefined at i=0 -> reuse frame 0 there
        # (const-vel then degenerates to persistence for that single step).
        z_prev2 = torch.cat([latents[:, :1], latents[:, :-2]], dim=1)
        return TeacherForced(pred=pred, target=target, z_prev=z_prev, z_prev2=z_prev2)

    @torch.no_grad()
    def rollout(self, latents: Tensor, actions: Tensor, states: Tensor, horizon: int) -> Tensor:
        """Open-loop. Seed with frame 0, grow the context with the model's own
        predictions (mirrors VJEPA2AC.rollout). Output h corresponds to real frame 1+h."""
        B, T, P, D = latents.shape
        horizon = min(horizon, T - 1)
        ctx = latents[:, :1]
        preds = []
        for _ in range(horizon):
            t = ctx.shape[1]
            y = self._predict(ctx.reshape(B, t * P, D), actions[:, :t], states[:, :t])
            nxt = y.view(B, t, P, D)[:, -1:]
            preds.append(nxt)
            ctx = torch.cat([ctx, nxt], dim=1)
        return torch.cat(preds, dim=1)                     # (B, horizon, P, D)

    def decode(self, latents: Tensor) -> Tensor:
        raise NotImplementedError("V-JEPA decoder not trained (Tier 4, out of scope)")

    def build_untrained(self) -> "VJEPA2ACAdapter":
        """Same frozen encoder, predictor re-initialised at random."""
        from evaluation.vjepa_ac.model import vit_ac_predictor

        rp = vit_ac_predictor(**_PRED_KWARGS).to(self.device).eval()
        for p in rp.parameters():
            p.requires_grad_(False)
        twin = copy.copy(self)
        twin._predictor = rp
        twin.name = f"{self.name}_untrained"
        twin.train_step = 0
        return twin
