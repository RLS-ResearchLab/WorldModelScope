"""DINO-WM -> WorldModelAdapter.

Frozen DINOv2 / EUPE patch encoder + a trained causal ViT latent predictor that
appends one action token per frame. Rebuilt from the config embedded in every
checkpoint; one class serves both encoders and names itself ``dino_wm_dinov2`` /
``dino_wm_eupe`` from ``model.encoder.type`` so their results don't collide.

Frame preprocessing matches training exactly: uint8 -> CHW -> ``/255`` in [0, 1],
no ImageNet normalisation (see ``datasets/_temp/dataset.py``).

Context lengths:
* teacher-forced -- the predictor's native ``num_hist`` (12) context frames.
* rollout        -- seeded with 3 real latent frames, then a sliding window of 3.
  ``ViTPredictor`` slices its positional embedding and causal mask to ``N`` tokens,
  so a shorter-than-``num_hist`` window is valid; this keeps the rollout horizon
  comparable to the other models on a 16-frame canonical clip.
"""
from __future__ import annotations

import copy

import torch
from torch import Tensor

from evaluation.common.adapter import TeacherForced
from models.world_models.factory import build_model
from src.utils.checkpoints import load_checkpoint

_ROLLOUT_SEED = 3


class DINOWMAdapter:
    action_dim = 7
    fps = 4.0
    is_action_conditioned = True
    has_decoder = True                 # trained decoders live in checkpoints/decoder/
    encoder_is_frozen = True
    num_context_frames = _ROLLOUT_SEED

    def __init__(self, ckpt: str, device: str = "cuda"):
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = raw["config"]
        enc_type = cfg["model"]["encoder"]["type"]

        self.device = device
        self.name = f"dino_wm_{enc_type}"
        self.encoder_fingerprint = f"dino_wm::{enc_type}"

        self.model = build_model(cfg).to(device).eval()
        info = load_checkpoint(ckpt, model=self.model, device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.train_step = int(info.get("step") if info.get("step") is not None else raw.get("step", -1))
        self.num_hist = int(self.model.num_hist)
        self.latent_dim = int(self.model.predictor.dim)
        self.tokens_per_frame = int(self.model.predictor.num_patches) - 1   # drop the action token

    # ---- encode / align ----
    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """(B, 16, 224, 224, 3) uint8 -> (B, 16, P, D)."""
        x = frames.permute(0, 1, 4, 2, 3).float().div_(255.0).to(self.device)
        return self.model.encode_observations(x)

    def align_actions(self, actions: Tensor) -> Tensor:
        """(B, 16, 7) pass-through: canonical ``actions[:, t]`` already drives the
        transition frame t -> t+1, which is DINO-WM's convention."""
        return actions.to(self.device)

    def align_states(self, states: Tensor) -> Tensor:
        return torch.zeros_like(states, device=self.device)    # unused by DINO-WM

    # ---- predictor calls ----
    @torch.no_grad()
    def teacher_forced(self, latents: Tensor, actions: Tensor, states: Tensor) -> TeacherForced:
        B, T, P, D = latents.shape
        H = min(self.num_hist, T - 1)
        pred = self.model.predict(latents[:, :H], actions[:, :H])          # preds of frames 1..H
        target = latents[:, 1:1 + H]
        z_prev = latents[:, :H]
        z_prev2 = torch.cat([latents[:, :1], latents[:, :H - 1]], dim=1)
        return TeacherForced(pred=pred, target=target, z_prev=z_prev, z_prev2=z_prev2, target_start=1)

    @torch.no_grad()
    def rollout(self, latents: Tensor, actions: Tensor, states: Tensor, horizon: int) -> Tensor:
        B, T, P, D = latents.shape
        C = self.num_context_frames
        horizon = min(horizon, T - C)
        ctx = latents[:, :C].clone()
        preds = []
        for h in range(horizon):
            nxt = self.model.predict(ctx, actions[:, h:h + C])[:, -1:]     # (B, 1, P, D)
            preds.append(nxt)
            ctx = torch.cat([ctx[:, 1:], nxt], dim=1)
        return torch.cat(preds, dim=1)

    def decode(self, latents: Tensor) -> Tensor:
        raise NotImplementedError("decoder lives in checkpoints/decoder/; Tier 4 is out of scope")

    def build_untrained(self) -> "DINOWMAdapter":
        """Same frozen encoder, predictor + action encoder re-initialised at random."""
        twin = copy.copy(self)
        m = copy.deepcopy(self.model)
        for mod in (m.predictor, m.action_encoder):
            for sub in mod.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
        if hasattr(m.predictor, "pos_embedding"):
            torch.nn.init.normal_(m.predictor.pos_embedding, std=0.02)
        m.to(self.device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        twin.model = m
        twin.name = f"{self.name}_untrained"
        twin.train_step = 0
        return twin
