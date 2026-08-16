"""import torch
import torch.nn as nn
import torch.nn.functional as F

class DINODecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        img_size=224,
        patch_size=14,
        decoder_dim=512,
        num_layers=6,
        num_heads=8,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.num_patches = (img_size // patch_size) ** 2

        self.input_proj = nn.Linear(
            latent_dim,
            decoder_dim
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches,
                decoder_dim
            )
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(
                decoder_dim,
                patch_size * patch_size * 3
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        z:
            [B, N, latent_dim]

        returns:
            [B, 3, H, W]
        """

        x = self.input_proj(z)

        x = x + self.pos_embed[:, :x.shape[1]]

        x = self.transformer(x)

        x = self.output_head(x)

        B, N, D = x.shape

        p = self.patch_size
        H = W = self.img_size // p

        x = x.reshape(
            B,
            H,
            W,
            p,
            p,
            3
        )

        x = x.permute(
            0, 5, 1, 3, 2, 4
        )

        x = x.reshape(
            B,
            3,
            self.img_size,
            self.img_size
        )

        return x"""