"""Pixel decoder for the V-JEPA 2.1-AC world model.

V-JEPA's tubelet-2 video encoder turns a 16-frame clip into 8 latent frames of
196 x 768 tokens. Each latent frame is decoded, independently, back to a
224 x 224 image with the same `PatchDecoder` used for the DINO-WM decoders
(196 = 14 x 14 patch grid, patch_size 16). Latent frame k is trained against
raw frame 2k (the first frame of its tubelet pair).

Only the decoder trains; the V-JEPA encoder is frozen. The RAE-style
noise-augmented-decoding trick from `EncoderDecoderModel` is reused verbatim so
the decoder also generalises to the *predictor's* slightly off-distribution
latents, not only clean encoder output.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.decoder.decoder_model import PatchDecoder

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_ENCODER = _REPO / "checkpoints/vjepa2_1_base/vjepa2_1_vitb_dist_vitG_384.pt"


def _build_vjepa_encoder(ckpt: str, device: str) -> nn.Module:
    """Frozen V-JEPA 2.1 ViT-B, constructed exactly as evaluation/vjepa_ac/model.py does."""
    from evaluation.vjepa_ac.model import CLIP_FRAMES, IMG_SIZE, TUBELET, _strip, vit_base

    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = _strip(raw.get("encoder") or raw.get("ema_encoder") or raw)
    enc = vit_base(
        patch_size=16, img_size=(IMG_SIZE, IMG_SIZE), num_frames=CLIP_FRAMES,
        tubelet_size=TUBELET, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
        uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
    )
    enc.load_state_dict(sd, strict=True)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc.to(device).eval()


class VJEPADecoderModel(nn.Module):
    """Trainer-facing wrapper: frozen V-JEPA encoder -> trainable per-frame PatchDecoder.

    Mirrors the interface of `models.decoder.decoder_builder.EncoderDecoderModel`
    (`compute_loss` / `validation_step` / `get_visualizations`, decoder-only
    `state_dict` / `load_state_dict`).
    """

    def __init__(self, config):
        super().__init__()
        from evaluation.vjepa_ac.model import CLIP_FRAMES, LATENT_FRAMES, TUBELET

        vcfg = config["vjepa"]
        self.device = config.get("training", {}).get("device", "cuda")
        enc_ckpt = vcfg.get("encoder_checkpoint") or str(_DEFAULT_ENCODER)
        self.encoder = _build_vjepa_encoder(enc_ckpt, self.device)

        self.clip_frames = CLIP_FRAMES         # 16
        self.tubelet = TUBELET                 # 2
        self.latent_frames = LATENT_FRAMES     # 8

        dc = config["decoder"]
        self.img_size = dc["img_size"]
        self.tokens_per_frame = (self.img_size // 16) ** 2   # 196
        self.latent_dim = 768
        self.decoder = PatchDecoder(
            latent_dim=self.latent_dim,
            num_patches=self.tokens_per_frame,
            img_size=dc["img_size"],
            patch_size=dc["patch_size"],
            decoder_dim=dc.get("decoder_dim", 768),
            num_layers=dc.get("num_layers", 8),
            num_heads=dc.get("num_heads", 8),
            mlp_ratio=dc.get("mlp_ratio", 4.0),
            dropout=dc.get("dropout", 0.0),
        ).to(self.device)

        self.l1_weight = config["training"].get("l1_weight", 0.1)
        self.noise_tau = dc.get("noise_tau", 0.0)

    # ---- decoder-only checkpointing ----
    def state_dict(self, *args, **kwargs):
        return self.decoder.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.decoder.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    # ---- core ----
    @torch.no_grad()
    def _encode(self, observations: torch.Tensor):
        """(B, 16, 3, H, W) in [0, 1]  ->  latents (B, 8, 196, 768), targets (B, 8, 3, 224, 224)."""
        B, T, C, H, W = observations.shape
        x = observations
        if (H, W) != (self.img_size, self.img_size):
            x = F.interpolate(
                x.reshape(B * T, C, H, W), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).reshape(B, T, C, self.img_size, self.img_size)
        video = ((x - 0.5) / 0.5).permute(0, 2, 1, 3, 4)         # (B, C, T, H, W) in [-1, 1]
        with torch.autocast(self.device, dtype=torch.bfloat16):
            z = self.encoder(video).float()                     # (B, 8*196, 768)
        z = z.view(B, self.latent_frames, self.tokens_per_frame, self.latent_dim)
        targets = x[:, ::self.tubelet]                           # (B, 8, C, H, W) -- frame 2k
        return z, targets

    def _augment_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # RAE-style noise-augmented decoding (see EncoderDecoderModel._augment_latents):
        # per-sample sigma ~ |N(0, tau^2)|, then latents += N(0, sigma^2), so the decoder
        # sees a range of noise levels and generalises off the clean-encoder manifold.
        if self.noise_tau <= 0:
            return latents
        sigma = torch.randn(latents.shape[0], 1, 1, device=latents.device,
                            dtype=latents.dtype).abs() * self.noise_tau
        return latents + torch.randn_like(latents) * sigma

    def _reconstruct(self, observations: torch.Tensor, add_noise: bool = False):
        z, targets = self._encode(observations)
        B, Fr, P, D = z.shape
        z = z.reshape(B * Fr, P, D)
        if add_noise:
            z = self._augment_latents(z)
        recon = self.decoder(z)                                  # (B*8, 3, 224, 224) in [0, 1]
        frames = targets.reshape(B * Fr, *targets.shape[2:])
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
