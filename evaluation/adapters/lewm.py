"""LeWM-bridge -> WorldModelAdapter.

ViT-S/14 encoder trained from scratch + an autoregressive AdaLN predictor, both
from the LeWM fork (`models/lewm/`). One CLS vector per frame, so ``P = 1``.

Rebuilt by hand from `checkpoints/lewm_bridge/config.json` -- no Hydra, no
`stable_worldmodel`. Two integration hurdles handled at import time:

* `stable_pretraining.backbone.utils.vit_hf` does ``from datasets import config``
  (HuggingFace). The repo's empty ``datasets/__init__.py`` shadows it, so the
  real one is imported first with the repo root off ``sys.path``.
* `models/lewm/` is put on the path for the fork's bare ``import jepa`` /
  ``import module``, then removed once the classes are bound.

Preprocessing matches training exactly (`models/lewm/train.py`):

* pixels -- ImageNet mean/std normalisation (not ``/255`` like DINO-WM).
* actions -- per-dim z-score with the BridgeData-train statistics from
  ``frames/bridge_train/action.npz``, then NaN -> 0.

Contexts: teacher-forced uses the predictor's native 12 frames; rollout seeds
with 3 real latent frames and slides a window of 3 (LeWM's own ``history_size``).
The encoder is trained (not frozen), so it differs at every checkpoint -- no
encode-once cache -- but ViT-S with only the CLS token is cheap.
"""
from __future__ import annotations

import copy
import functools
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from evaluation.common.adapter import TeacherForced

_REPO = Path(__file__).resolve().parents[2]
_LEWM = _REPO / "models" / "lewm"
_ACTION_NPZ = _REPO / "frames" / "bridge_train" / "action.npz"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_ROLLOUT_SEED = 3
_NUM_HIST = 12
_EMBED_DIM = 384

# ---- imports that need the path juggling ----
_dropped = [p for p in ("", ".", str(_REPO)) if p in sys.path]
for _p in _dropped:
    sys.path.remove(_p)
sys.path.insert(0, str(_LEWM))
try:
    import datasets as _hf_datasets  # noqa: F401  -- real HuggingFace datasets into sys.modules
    from stable_pretraining.backbone.utils import vit_hf
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP
finally:
    if str(_LEWM) in sys.path:
        sys.path.remove(str(_LEWM))
    for _p in _dropped:
        sys.path.insert(0, _p)


def _build_jepa() -> JEPA:
    enc = vit_hf(size="small", patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
    pred = ARPredictor(
        num_frames=_NUM_HIST, input_dim=_EMBED_DIM, hidden_dim=_EMBED_DIM, output_dim=_EMBED_DIM,
        depth=16, heads=6, mlp_dim=1536, dim_head=64, dropout=0.1, emb_dropout=0.0,
    )
    ae = Embedder(input_dim=7, emb_dim=_EMBED_DIM)
    bn = functools.partial(nn.BatchNorm1d)
    proj = MLP(_EMBED_DIM, 2048, _EMBED_DIM, norm_fn=bn)
    pred_proj = MLP(_EMBED_DIM, 2048, _EMBED_DIM, norm_fn=bn)
    return JEPA(enc, pred, ae, proj, pred_proj)


class LeWMAdapter:
    name = "lewm_bridge"
    latent_dim = _EMBED_DIM
    tokens_per_frame = 1                # CLS token only
    num_context_frames = _ROLLOUT_SEED
    action_dim = 7
    fps = 4.0                           # canonical; LeWM trained at 5 fps (frameskip 1)
    is_action_conditioned = True
    has_decoder = False
    encoder_is_frozen = False           # trained from scratch -> differs per checkpoint

    def __init__(self, ckpt: str, device: str = "cuda"):
        self.device = device
        self.model = _build_jepa()
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"LeWM state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.train_step = int(Path(ckpt).stem.split("_")[-1])
        self.encoder_fingerprint = f"lewm_bridge::{Path(ckpt).name}"   # unique per ckpt
        self.num_hist = _NUM_HIST

        # z-score stats, exactly as models/lewm/utils.get_column_normalizer
        arr = torch.from_numpy(np.load(_ACTION_NPZ)["arr_0"]).float()
        arr = arr[~torch.isnan(arr).any(dim=1)]
        self._a_mean = arr.mean(0).to(device)
        self._a_std = arr.std(0).clamp_min(1e-6).to(device)
        self._px_mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 1, 3, 1, 1)
        self._px_std = torch.tensor(_IMAGENET_STD, device=device).view(1, 1, 3, 1, 1)

    # ---- encode / align ----
    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """(B, 16, 224, 224, 3) uint8 -> (B, 16, 1, 384)."""
        x = frames.permute(0, 1, 4, 2, 3).float().div_(255.0).to(self.device)
        x = (x - self._px_mean) / self._px_std
        b, t = x.shape[:2]
        x = x.reshape(b * t, *x.shape[2:])
        out = self.model.encoder(x, interpolate_pos_encoding=True)
        emb = self.model.projector(out.last_hidden_state[:, 0])        # (b*t, 384)
        return emb.reshape(b, t, 1, self.latent_dim)

    def align_actions(self, actions: Tensor) -> Tensor:
        """(B, 16, 7) -> z-scored (B, 16, 7); NaN -> 0."""
        a = torch.nan_to_num(actions.to(self.device), 0.0)
        return (a - self._a_mean) / self._a_std

    def align_states(self, states: Tensor) -> Tensor:
        return torch.zeros_like(states, device=self.device)

    # ---- predictor calls ----
    @torch.no_grad()
    def _predict(self, emb: Tensor, actions_z: Tensor) -> Tensor:
        """(B, T, D) + (B, T, 7) -> (B, T, D)."""
        act_emb = self.model.action_encoder(actions_z)
        return self.model.predict(emb, act_emb)

    @torch.no_grad()
    def teacher_forced(self, latents: Tensor, actions: Tensor, states: Tensor) -> TeacherForced:
        B, T, P, D = latents.shape
        emb = latents[:, :, 0]                                        # (B, T, D)
        H = min(self.num_hist, T - 1)
        pred = self._predict(emb[:, :H], actions[:, :H])[:, :, None]  # (B, H, 1, D) -> frames 1..H
        target = latents[:, 1:1 + H]
        z_prev = latents[:, :H]
        z_prev2 = torch.cat([latents[:, :1], latents[:, :H - 1]], dim=1)
        return TeacherForced(pred=pred, target=target, z_prev=z_prev, z_prev2=z_prev2, target_start=1)

    @torch.no_grad()
    def rollout(self, latents: Tensor, actions: Tensor, states: Tensor, horizon: int) -> Tensor:
        B, T, P, D = latents.shape
        C = self.num_context_frames
        horizon = min(horizon, T - C)
        emb = latents[:, :C, 0].clone()                               # (B, C, D)
        preds = []
        for h in range(horizon):
            nxt = self._predict(emb[:, -C:], actions[:, h:h + C])[:, -1:]   # (B, 1, D)
            preds.append(nxt)
            emb = torch.cat([emb, nxt], dim=1)
        return torch.cat(preds, dim=1)[:, :, None]                    # (B, horizon, 1, D)

    def decode(self, latents: Tensor) -> Tensor:
        raise NotImplementedError("LeWM CLS->image inversion not trained (Tier 4, out of scope)")

    def build_untrained(self) -> "LeWMAdapter":
        """Same trained encoder, predictor + action encoder + pred_proj re-initialised."""
        twin = copy.copy(self)
        m = copy.deepcopy(self.model)
        for mod in (m.predictor, m.action_encoder, m.pred_proj):
            for sub in mod.modules():
                if hasattr(sub, "reset_parameters"):
                    sub.reset_parameters()
        if hasattr(m.predictor, "pos_embedding"):
            nn.init.normal_(m.predictor.pos_embedding, std=0.02)
        m.to(self.device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        twin.model = m
        twin.name = f"{self.name}_untrained"
        twin.train_step = 0
        return twin
