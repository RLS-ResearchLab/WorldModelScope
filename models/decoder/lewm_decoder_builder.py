"""Pixel decoder for the LeWM-bridge world model.

LeWM summarises each frame as a single projected CLS vector (384-d,
``tokens_per_frame = 1``), so unlike the DINO-WM / V-JEPA decoders there is no
patch grid for ``PatchDecoder`` to invert. This wraps the frozen LeWM encoder
(ViT-S/14 + BatchNorm projector, rebuilt exactly as
``evaluation/adapters/lewm.py``) and trains a ``GlobalLatentDecoder``: learned
position tokens AdaLN-conditioned on that one vector -> 224x224 image.

Only the decoder trains. The RAE-style noise-augmented-decoding trick from
``EncoderDecoderModel`` is reused so the decoder also generalises to the
*predictor's* slightly off-distribution vectors, not only clean encoder output.

Single-vector inversion is genuinely lossy -- expect blurrier reconstructions
than DINO/V-JEPA. That gap is the informative Tier-4 signal: it quantifies how
much spatial detail LeWM's 1-token bottleneck discards.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.decoder_model import GlobalLatentDecoder

_REPO = Path(__file__).resolve().parents[2]
_LEWM = _REPO / "models" / "lewm"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_EMBED_DIM = 384
_NUM_HIST = 12


_LEWM_STACK = None


def _import_lewm_stack():
    """Import LeWM's model classes past the ``datasets`` namespace clash.

    The repo's empty ``datasets/__init__.py`` shadows HuggingFace ``datasets``,
    which ``stable_pretraining`` needs. ``train_decoder.py`` imports the local
    ``datasets.dataloader`` first, so it is already cached -- pop it, import the
    LeWM stack against the real HF ``datasets`` (repo root off ``sys.path``),
    then restore the local package so ``build_dataloaders`` keeps working.

    Cached after the first call: re-importing HF ``datasets`` a second time in the
    same process re-registers pyarrow extension types and raises.
    """
    global _LEWM_STACK
    if _LEWM_STACK is not None:
        return _LEWM_STACK

    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "datasets" or k.startswith("datasets.")
    }
    dropped = [p for p in ("", ".", str(_REPO)) if p in sys.path]
    for p in dropped:
        sys.path.remove(p)
    sys.path.insert(0, str(_LEWM))
    try:
        import datasets as _hf  # noqa: F401  -- real HuggingFace datasets into sys.modules
        from stable_pretraining.backbone.utils import vit_hf
        from jepa import JEPA
        from module import ARPredictor, Embedder, MLP

        _LEWM_STACK = (vit_hf, JEPA, ARPredictor, Embedder, MLP)
        return _LEWM_STACK
    finally:
        if str(_LEWM) in sys.path:
            sys.path.remove(str(_LEWM))
        for p in dropped:
            sys.path.insert(0, p)
        # stable_pretraining now holds its own references to the HF datasets
        # internals it needs, so hand sys.modules['datasets'] back to the local
        # package for the rest of the process (build_dataloaders, etc.).
        for k in list(sys.modules):
            if k == "datasets" or k.startswith("datasets."):
                del sys.modules[k]
        sys.modules.update(saved)


def _build_jepa():
    """LeWM JEPA with random weights, matching checkpoints/lewm_bridge/config.json."""
    vit_hf, JEPA, ARPredictor, Embedder, MLP = _import_lewm_stack()
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


class LeWMDecoderModel(nn.Module):
    """Trainer-facing wrapper: frozen LeWM encoder+projector -> trainable GlobalLatentDecoder.

    Mirrors the interface of ``models.decoder.decoder_builder.EncoderDecoderModel``
    (``compute_loss`` / ``validation_step`` / ``get_visualizations``, decoder-only
    ``state_dict`` / ``load_state_dict``).
    """

    def __init__(self, config):
        super().__init__()
        lcfg = config["lewm"]
        self.device = config.get("training", {}).get("device", "cuda")

        jepa = _build_jepa()
        sd = torch.load(lcfg["encoder_checkpoint"], map_location="cpu", weights_only=True)
        missing, unexpected = jepa.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"LeWM encoder state_dict: unexpected keys {unexpected[:5]}")
        # Only the encoder + projector matter; the projected CLS is exactly what the adapter's
        # encode() (and therefore the predictor, and Tier 4) operates on.
        self.encoder = jepa.encoder.to(self.device).eval()
        self.projector = jepa.projector.to(self.device).eval()
        for p in (*self.encoder.parameters(), *self.projector.parameters()):
            p.requires_grad_(False)

        self.latent_dim = _EMBED_DIM
        dc = config["decoder"]
        self.img_size = dc["img_size"]
        self.decoder = GlobalLatentDecoder(
            latent_dim=self.latent_dim,
            img_size=dc["img_size"],
            patch_size=dc["patch_size"],
            decoder_dim=dc.get("decoder_dim", 512),
            num_layers=dc.get("num_layers", 8),
            num_heads=dc.get("num_heads", 8),
            mlp_ratio=dc.get("mlp_ratio", 4.0),
            dropout=dc.get("dropout", 0.0),
        ).to(self.device)

        self.l1_weight = config["training"].get("l1_weight", 0.1)
        self.noise_tau = dc.get("noise_tau", 0.0)
        self._px_mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._px_std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    # ---- decoder-only checkpointing ----
    def state_dict(self, *args, **kwargs):
        return self.decoder.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.decoder.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.projector.eval()
        return self

    # ---- core ----
    def _prepare_frames(self, observations: torch.Tensor):
        B, T, C, H, W = observations.shape
        x = observations.reshape(B * T, C, H, W)
        if (H, W) != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        return x

    @torch.no_grad()
    def _encode(self, observations: torch.Tensor):
        """(B, T, 3, H, W) in [0, 1]  ->  latents (B*T, 384), targets (B*T, 3, img, img) in [0, 1]."""
        frames = self._prepare_frames(observations)
        x = (frames - self._px_mean) / self._px_std
        out = self.encoder(x, interpolate_pos_encoding=True)
        z = self.projector(out.last_hidden_state[:, 0]).float()   # (B*T, 384)
        return z, frames

    def _augment_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # RAE-style noise-augmented decoding (see EncoderDecoderModel._augment_latents):
        # per-sample sigma ~ |N(0, tau^2)|, then latents += N(0, sigma^2).
        if self.noise_tau <= 0:
            return latents
        sigma = torch.randn(latents.shape[0], 1, device=latents.device,
                            dtype=latents.dtype).abs() * self.noise_tau
        return latents + torch.randn_like(latents) * sigma

    def _reconstruct(self, observations: torch.Tensor, add_noise: bool = False):
        z, frames = self._encode(observations)
        if add_noise:
            z = self._augment_latents(z)
        recon = self.decoder(z)                                   # (B*T, 3, img, img) in [0, 1]
        return frames, recon

    def _step(self, batch, add_noise: bool = False):
        frames, recon = self._reconstruct(batch["observations"], add_noise=add_noise)
        loss = F.mse_loss(recon, frames) + self.l1_weight * F.l1_loss(recon, frames)
        return loss, {"recon_loss": loss.detach()}

    def compute_loss(self, batch):
        return self._step(batch, add_noise=True)

    @torch.no_grad()
    def validation_step(self, batch):
        return self._step(batch, add_noise=False)

    @torch.no_grad()
    def get_visualizations(self, batch, max_images: int = 8):
        frames, recon = self._reconstruct(batch["observations"])
        return {
            "ground_truth": frames[:max_images].detach().cpu(),
            "reconstruction": recon[:max_images].detach().cpu(),
        }
