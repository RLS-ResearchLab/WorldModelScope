import torch
import torch.nn as nn
from einops import rearrange


class PatchDecoder(nn.Module):
    """Decodes a sequence of patch-token latents [B, N, D] back into an image [B, C, H, W].

    N must equal (img_size // patch_size) ** 2 -- i.e. num_patches must match the token grid
    the upstream encoder actually produces for this img_size.
    """

    def __init__(
        self,
        latent_dim,
        num_patches,
        img_size,
        patch_size,
        out_channels=3,
        decoder_dim=384,
        num_layers=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()

        grid_size = img_size // patch_size
        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"num_patches ({num_patches}) does not match the img_size/patch_size grid "
                f"({grid_size}x{grid_size}={grid_size * grid_size})."
            )

        self.num_patches = num_patches
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.input_proj = nn.Linear(latent_dim, decoder_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, decoder_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(decoder_dim)

        self.pixel_head = nn.Linear(decoder_dim, patch_size * patch_size * out_channels)

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, latents):
        B, N, D = latents.shape
        if N != self.num_patches:
            raise ValueError(f"PatchDecoder built for {self.num_patches} tokens, got {N}.")

        x = self.input_proj(latents) + self.pos_embedding
        x = self.norm(self.transformer(x))
        patches = self.pixel_head(x)

        image = rearrange(
            patches, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.grid_size, w=self.grid_size,
            p1=self.patch_size, p2=self.patch_size, c=self.out_channels,
        )
        image = image + self.refine(image)
        return torch.sigmoid(image)
