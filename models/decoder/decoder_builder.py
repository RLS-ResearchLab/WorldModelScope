import torch
import torch.nn as nn
import torch.nn.functional as F

from models.world_models.factory import build_model
from models.decoder.decoder_model import PatchDecoder


def build_decoder_pipeline(config):
    """Builds a frozen DINOWM (for its encoder+feature_adapter) and a trainable PatchDecoder
    sized to match the exact token grid/dim that encode_observations() produces."""
    dino_wm_cfg = config["dino_wm"]
    dino_wm = build_model(dino_wm_cfg)
    dino_wm.eval()
    for p in dino_wm.parameters():
        p.requires_grad = False

    if config.get("dino_wm_checkpoint"):
        from src.utils.checkpoints import load_checkpoint
        load_checkpoint(config["dino_wm_checkpoint"], model=dino_wm)

    img_size = dino_wm_cfg["model"]["image_size"]
    probe_device = next(dino_wm.parameters()).device
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=probe_device)
        latents = dino_wm.encode_observations(dummy.unsqueeze(1))  # [1, 1, P, D]
    num_patches, latent_dim = latents.shape[2], latents.shape[3]

    decoder_cfg = config["decoder"]
    decoder = PatchDecoder(
        latent_dim=latent_dim,
        num_patches=num_patches,
        img_size=decoder_cfg["img_size"],
        patch_size=decoder_cfg["patch_size"],
        decoder_dim=decoder_cfg.get("decoder_dim", 384),
        num_layers=decoder_cfg.get("num_layers", 6),
        num_heads=decoder_cfg.get("num_heads", 6),
        mlp_ratio=decoder_cfg.get("mlp_ratio", 4.0),
        dropout=decoder_cfg.get("dropout", 0.0),
    )
    decoder = decoder.to(probe_device)

    return dino_wm, decoder


class EncoderDecoderModel(nn.Module):
    """Trainer-facing wrapper: frozen DINOWM encoder -> trainable PatchDecoder, reconstruction
    loss against the source frame. Only the decoder is trained/saved."""

    def __init__(self, config):
        super().__init__()
        self.dino_wm, self.decoder = build_decoder_pipeline(config)
        self.img_size = config["decoder"]["img_size"]
        self.l1_weight = config["training"].get("l1_weight", 0.1)
        self.noise_tau = config["decoder"].get("noise_tau", 0.0)

    def state_dict(self, *args, **kwargs):
        return self.decoder.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.decoder.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.dino_wm.eval()
        return self

    def _prepare_frames(self, observations):
        B, T, C, H, W = observations.shape
        frames = observations.reshape(B * T, C, H, W)
        return F.interpolate(frames, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

    def _reconstruct(self, observations, add_noise=False):
        with torch.no_grad():
            latents = self.dino_wm.encode_observations(observations)  # [B, T, P, D]
        B, T, P, D = latents.shape
        frames = self._prepare_frames(observations)
        latents = latents.reshape(B * T, P, D)
        if add_noise:
            latents = self._augment_latents(latents)
        recon = self.decoder(latents)
        return frames, recon

    def _augment_latents(self, latents):
        # RAE-style noise-augmented decoding (Zheng et al., "Diffusion Transformers with
        # Representation Autoencoders"): smooth the discrete training latents with small
        # additive Gaussian noise so the decoder generalizes to slightly off-distribution
        # latents (e.g. from a trained predictor) instead of only ever seeing clean encoder
        # output. sigma is itself resampled per-batch from a half-normal so the decoder sees
        # a range of noise levels rather than one fixed scale.
        if self.noise_tau <= 0:
            return latents
        sigma = torch.randn(latents.shape[0], 1, 1, device=latents.device, dtype=latents.dtype).abs() * self.noise_tau
        return latents + torch.randn_like(latents) * sigma

    def _step(self, batch, add_noise=False):
        frames, recon = self._reconstruct(batch["observations"], add_noise=add_noise)
        loss = F.mse_loss(recon, frames) + self.l1_weight * F.l1_loss(recon, frames)
        return loss, {"recon_loss": loss.detach()}

    def compute_loss(self, batch):
        return self._step(batch, add_noise=True)

    @torch.no_grad()
    def validation_step(self, batch):
        return self._step(batch, add_noise=False)

    @torch.no_grad()
    def get_visualizations(self, batch, max_images=8):
        frames, recon = self._reconstruct(batch["observations"])
        return {
            "ground_truth": frames[:max_images].detach().cpu(),
            "reconstruction": recon[:max_images].detach().cpu(),
        }
