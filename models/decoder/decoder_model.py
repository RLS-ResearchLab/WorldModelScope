
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders.dinov2 import DINOv2Encoder
from models.encoders.eupe_encoder import EUPEEncoder
from models.decoder import ViTDecoder

def build_encoder(config):
    encoder_cfg = config["encoder"]
    name = encoder_cfg["name"].lower()

    if name == "dinov2":
        return DINOv2Encoder(
            model_name=encoder_cfg.get("variant", "dinov2_vits14"),
        )

    if name == "eupe":
        return EUPEEncoder()

    raise ValueError(
        f"Unknown encoder '{name}'. Expected 'dinov2' or 'eupe'."
    )


def build_decoder(config):
    model_cfg = config["model"]

    return ViTDecoder(
        latent_dim=model_cfg["latent_dim"],
        num_patches=model_cfg["num_patches"],
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        decoder_dim=model_cfg.get("decoder_dim", 384),
        num_layers=model_cfg.get("num_layers", 6),
        num_heads=model_cfg.get("num_heads", 6),
        dim_head=model_cfg.get("dim_head", 64),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        out_channels=model_cfg.get("out_channels", 3),
        dropout=model_cfg.get("dropout", 0.0),
        use_conv_refinement=model_cfg.get("use_conv_refinement", True),
        refinement_hidden=model_cfg.get("refinement_hidden", 64),
        refinement_blocks=model_cfg.get("refinement_blocks", 3),
    )

class DecoderModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = build_encoder(config)
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = build_decoder(config)

        self.img_size = config["model"]["img_size"]
        self.l1_weight = config["training"].get("l1_weight", 0.1)

    def state_dict(self, *args, **kwargs):
        return self.decoder.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        return self.decoder.load_state_dict(state_dict, *args, **kwargs)

    def train(self, mode=True):
        # Keep the frozen encoder in eval mode regardless of what the
        # Trainer does with the wrapper (it calls model.train() every epoch).
        super().train(mode)
        self.encoder.eval()
        return self

    # --------------------------------------------------------------
    def _prepare_frames(self, observations):
        
        B, T, C, H, W = observations.shape
        frames = observations.reshape(B * T, C, H, W)

        frames = F.interpolate(
            frames,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        return frames

    def _step(self, batch):
        frames = self._prepare_frames(batch["observations"])

        with torch.no_grad():
            latents = self.encoder(frames)

        reconstruction = self.decoder(latents)

        mse = F.mse_loss(reconstruction, frames)
        l1 = F.l1_loss(reconstruction, frames)
        loss = mse + self.l1_weight * l1

        metrics = {
            "mse": mse.detach(),
            "l1": l1.detach(),
        }

        return loss, metrics

    def compute_loss(self, batch):
        return self._step(batch)

    @torch.no_grad()
    def validation_step(self, batch):
        return self._step(batch)