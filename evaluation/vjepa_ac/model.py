"""V-JEPA 2.1 ViT-B + action-conditioned predictor, assembled for evaluation.

Mirrors exactly the model construction and preprocessing in the training script
`scripts/11_train_vjepa21_base_ac.py` of the VJEPA2P1_ac repo:

  encoder   : app.vjepa_2_1.models.vision_transformer.vit_base  (frozen)
  predictor : src.models.ac_predictor.vit_ac_predictor           (trained)

The reference V-JEPA 2.1 source tree is vendored under `_vendor/vjepa2`
(pinned commit in `_vendor/VJEPA2_COMMIT.txt`). Override with VJEPA2_SRC if you
want to point at a different checkout.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_VENDORED = Path(__file__).resolve().parent / "_vendor" / "vjepa2"
VJEPA2_SRC = os.environ.get("VJEPA2_SRC") or str(_VENDORED)
if VJEPA2_SRC not in sys.path:
    sys.path.insert(0, VJEPA2_SRC)

from app.vjepa_2_1.models.vision_transformer import vit_base  # noqa: E402
from src.models.ac_predictor import vit_ac_predictor          # noqa: E402

# Clip contract, fixed by the trained checkpoint.
CLIP_FRAMES = 16          # raw frames fed to the encoder
TUBELET = 2               # encoder temporal patch -> 8 latent frames
LATENT_FRAMES = CLIP_FRAMES // TUBELET   # 8
IMG_SIZE = 224
ACTION_DIM = 7
STATE_DIM = 7


def _strip(sd: dict) -> dict:
    return {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}


class VJEPA2AC(nn.Module):
    def __init__(self, encoder_ckpt: str, predictor_ckpt: str, device: str = "cuda"):
        super().__init__()
        self.device = device

        enc_raw = torch.load(encoder_ckpt, map_location="cpu", weights_only=True)
        enc_sd = _strip(enc_raw.get("encoder") or enc_raw.get("ema_encoder") or enc_raw)
        self.encoder = vit_base(
            patch_size=16, img_size=(IMG_SIZE, IMG_SIZE), num_frames=CLIP_FRAMES,
            tubelet_size=TUBELET, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
            uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
        )
        self.encoder.load_state_dict(enc_sd, strict=True)

        self.predictor = vit_ac_predictor(
            img_size=IMG_SIZE, patch_size=16, num_frames=LATENT_FRAMES, tubelet_size=1,
            embed_dim=768, predictor_embed_dim=768, depth=12, num_heads=12,
            use_rope=False, use_activation_checkpointing=False,
        )
        pred_ckpt = torch.load(predictor_ckpt, map_location="cpu", weights_only=False)
        pred_sd = pred_ckpt["predictor"] if "predictor" in pred_ckpt else pred_ckpt
        self.predictor.load_state_dict(_strip(pred_sd), strict=True)
        self.train_step = int(pred_ckpt.get("step", -1)) if isinstance(pred_ckpt, dict) else -1

        for p in self.parameters():
            p.requires_grad_(False)
        self.to(device).eval()
        self.tokens_per_frame = (IMG_SIZE // 16) ** 2  # 196

        self.param_counts = {
            "encoder": sum(p.numel() for p in self.encoder.parameters()),
            "predictor": sum(p.numel() for p in self.predictor.parameters()),
        }

    # ---- preprocessing (identical to the training script's `batch()`) ----
    @staticmethod
    def preprocess_frames(frames_uint8: torch.Tensor) -> torch.Tensor:
        """[B, 16, H, W, 3] uint8  ->  [B, 3, 16, 224, 224] float in [-1, 1]."""
        x = frames_uint8.float() / 255.0
        x = (x - 0.5) / 0.5
        v = x.permute(0, 4, 1, 2, 3)                       # B,C,T,H,W
        B, C, T = v.shape[:3]
        v = F.interpolate(
            v.permute(0, 2, 1, 3, 4).flatten(0, 1), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        ).view(B, T, C, IMG_SIZE, IMG_SIZE).permute(0, 2, 1, 3, 4)
        return v

    # ---- core ops ----
    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """[B, 3, 16, 224, 224] -> latent tokens [B, 8*196, 768]."""
        with torch.autocast(self.device, dtype=torch.bfloat16):
            return self.encoder(video.to(self.device)).float()

    @torch.no_grad()
    def predict(self, z: torch.Tensor, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Latent tokens + per-latent-frame actions/states -> predicted latents.

        Returns y with the same shape as z; y[:, :-K] is the prediction of the
        latent at frame t+1 given the frame-causal context up to t, aligned with
        z[:, K:] (K = tokens_per_frame).
        """
        with torch.autocast(self.device, dtype=torch.bfloat16):
            return self.predictor(z.to(self.device), actions.to(self.device), states.to(self.device)).float()

    @torch.no_grad()
    def teacher_forced(self, video, actions, states):
        """One-step next-latent metrics + the copy-previous-frame baseline.

        Returns batch-mean scalars plus `clip_smooth_l1`, a per-clip list for
        distribution plots.
        """
        z = self.encode(video)
        y = self.predict(z, actions, states)
        k = self.tokens_per_frame
        pred, tgt = y[:, :-k], z[:, k:]
        ident = z[:, :-k]                                  # "next frame == current frame"
        per_clip = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=(1, 2))
        return {
            "smooth_l1": F.smooth_l1_loss(pred, tgt).item(),
            "l1": F.l1_loss(pred, tgt).item(),
            "mse": F.mse_loss(pred, tgt).item(),
            "cos_sim": F.cosine_similarity(pred, tgt, dim=-1).mean().item(),
            "identity_smooth_l1": F.smooth_l1_loss(ident, tgt).item(),
            "identity_l1": F.l1_loss(ident, tgt).item(),
            "clip_smooth_l1": per_clip.cpu().tolist(),
        }

    @torch.no_grad()
    def rollout(self, video, actions, states):
        """Open-loop latent rollout: seed with frame-0 latents, then feed the
        model its own predictions. Returns per-horizon latent L1 (h = 1..7)."""
        z = self.encode(video)
        k = self.tokens_per_frame
        T = LATENT_FRAMES
        z_frames = z.view(z.shape[0], T, k, z.shape[-1])
        ctx = z_frames[:, :1]                              # [B, 1, k, D]
        out = {}
        for h in range(1, T):
            t = ctx.shape[1]
            y = self.predict(ctx.flatten(1, 2), actions[:, :t], states[:, :t])
            nxt = y.view(y.shape[0], t, k, y.shape[-1])[:, -1:]   # predicted frame t
            out[f"rollout_l1_h{h}"] = F.l1_loss(nxt, z_frames[:, h:h + 1]).item()
            ctx = torch.cat([ctx, nxt], dim=1)
        return out
