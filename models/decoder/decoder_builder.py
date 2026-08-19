
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.dinov2 import DINOv2Encoder
from models.encoders.EUPE_encoder import EUPEEncoder
from models.decoder.decoder_model import ViTDecoder

ENCODER_REGISTRY = {
    "dinov2": DINOv2Encoder,
    "eupe": EUPEEncoder,
}

def build_encoder(config):
    encoder_cfg = config["encoder"]
    name = encoder_cfg["name"].lower()
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Options: {list(ENCODER_REGISTRY)}")
    cls = ENCODER_REGISTRY[name]
    kwargs = {k: v for k, v in encoder_cfg.items() if k != "name"}
    return cls(**kwargs)

class LatentPipeline(nn.Module):
    def __init__(self, encoder, predictor=None):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, frames):
        encoder_latents = self.encoder(frames)
        predictor_latents = None
        if self.predictor is not None:
            predictor_latents = self.predictor(encoder_latents)
        return encoder_latents, predictor_latents

def build_decoders(config, latent_pipeline):
    img_size = config["model"]["img_size"]
    dummy = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        encoder_latents, predictor_latents = latent_pipeline(dummy)

    decoder_for_encoder = ViTDecoder(config, *infer_latent_shape(encoder_latents))

    decoder_for_predictor = None
    if predictor_latents is not None:
        decoder_for_predictor = ViTDecoder(config, *infer_latent_shape(predictor_latents))

    return decoder_for_encoder, decoder_for_predictor

def infer_latent_shape(latents):
    """latents: [B, N, D] -> (num_patches, latent_dim)"""
    _, num_patches, latent_dim = latents.shape
    return num_patches, latent_dim

class DecoderModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        encoder = build_encoder(config)
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

        predictor = build_predictor(config)  # can be None
        self._predictor_frozen = True
        if predictor is not None:
            self._predictor_frozen = config["predictor"].get("frozen", True)
            if self._predictor_frozen:
                predictor.eval()
                for p in predictor.parameters():
                    p.requires_grad = False

        self.latent_pipeline = LatentPipeline(encoder, predictor)

        self.decoder_for_encoder, self.decoder_for_predictor = build_decoders(
            config, self.latent_pipeline
        )

        self.img_size = config["model"]["img_size"]
        self.l1_weight = config["training"].get("l1_weight", 0.1)

    def state_dict(self, *args, **kwargs):
        sd = {"decoder_for_encoder": self.decoder_for_encoder.state_dict(*args, **kwargs)}
        if self.decoder_for_predictor is not None:
            sd["decoder_for_predictor"] = self.decoder_for_predictor.state_dict(*args, **kwargs)
        return sd

    def load_state_dict(self, state_dict, *args, **kwargs):
        self.decoder_for_encoder.load_state_dict(state_dict["decoder_for_encoder"], *args, **kwargs)
        if self.decoder_for_predictor is not None and "decoder_for_predictor" in state_dict:
            self.decoder_for_predictor.load_state_dict(state_dict["decoder_for_predictor"], *args, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.latent_pipeline.encoder.eval()
        if self.latent_pipeline.predictor is not None and self._predictor_frozen:
            self.latent_pipeline.predictor.eval()
        return self

    def _prepare_frames(self, observations):
        B, T, C, H, W = observations.shape
        frames = observations.reshape(B * T, C, H, W)
        frames = F.interpolate(
            frames, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        return frames

    def _reconstruct(self, frames):
        """Run the full pipeline and return everything: frames + both reconstructions.
        This is the one place that actually produces images, not just losses."""
        encoder_latents, predictor_latents = self.latent_pipeline(frames)

        recon_from_encoder = self.decoder_for_encoder(encoder_latents)

        recon_from_predictor = None
        if predictor_latents is not None:
            recon_from_predictor = self.decoder_for_predictor(predictor_latents)

        return {
            "frames": frames,                          # ground-truth images
            "recon_from_encoder": recon_from_encoder,   # decoded encoder latents
            "recon_from_predictor": recon_from_predictor,  # decoded predictor latents (or None)
        }

    def _step(self, batch):
        frames = self._prepare_frames(batch["observations"])
        outputs = self._reconstruct(frames)

        recon_from_encoder = outputs["recon_from_encoder"]
        loss_encoder = F.mse_loss(recon_from_encoder, frames) \
                     + self.l1_weight * F.l1_loss(recon_from_encoder, frames)

        metrics = {"loss_encoder": loss_encoder.detach()}
        total_loss = loss_encoder

        recon_from_predictor = outputs["recon_from_predictor"]
        if recon_from_predictor is not None:
            loss_predictor = F.mse_loss(recon_from_predictor, frames) \
                            + self.l1_weight * F.l1_loss(recon_from_predictor, frames)
            metrics["loss_predictor"] = loss_predictor.detach()
            total_loss = total_loss + loss_predictor

        return total_loss, metrics

    def compute_loss(self, batch):
        return self._step(batch)

    @torch.no_grad()
    def validation_step(self, batch):
        return self._step(batch)

    @torch.no_grad()
    def get_visualizations(self, batch, max_images=8):
        """Call this from the training loop/logger periodically (e.g. every N steps
        or once per epoch) to get actual image tensors for logging — NOT every
        step, since it's redundant with _step's own forward pass and you don't
        want to double the compute cost on every iteration just to save images."""
        frames = self._prepare_frames(batch["observations"])
        outputs = self._reconstruct(frames)

        return {
            "ground_truth": outputs["frames"][:max_images].detach().cpu(),
            "recon_from_encoder": outputs["recon_from_encoder"][:max_images].detach().cpu(),
            "recon_from_predictor": (
                outputs["recon_from_predictor"][:max_images].detach().cpu()
                if outputs["recon_from_predictor"] is not None else None
            ),
        }