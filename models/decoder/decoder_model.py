import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _modulate(x, shift, scale):
    """AdaLN-zero modulation over a token sequence; shift/scale are [B, D]."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _AdaLNBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning on a single global vector (DiT / LeWM
    ``ConditionalBlock`` style): the condition sets per-layer shift/scale/gate, and the gate
    is zero-initialised so the block starts as identity."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.ada(c).chunk(6, dim=-1)
        h = _modulate(self.norm1(x), shift_a, scale_a)
        x = x + gate_a.unsqueeze(1) * self.attn(h, h, h, need_weights=False)[0]
        h = _modulate(self.norm2(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(h)
        return x


class GlobalLatentDecoder(nn.Module):
    """Decodes a single global latent vector [B, D] (or [B, 1, D]) into an image [B, C, H, W].

    For encoders that summarise a whole frame as one vector -- e.g. LeWM's projected CLS
    token -- so there is no patch grid for ``PatchDecoder`` to invert. Learned position
    tokens are AdaLN-conditioned on the latent, passed through a small transformer, then
    projected to pixels with the same patch-head + conv-refine + sigmoid tail as
    ``PatchDecoder``. ``patch_size`` here is purely a decoder-internal output grid, not tied
    to any encoder.
    """

    def __init__(
        self,
        latent_dim,
        img_size,
        patch_size,
        out_channels=3,
        decoder_dim=512,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        grid_size = img_size // patch_size
        if grid_size * patch_size != img_size:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")

        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.cond = nn.Sequential(
            nn.Linear(latent_dim, decoder_dim), nn.SiLU(), nn.Linear(decoder_dim, decoder_dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, decoder_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [_AdaLNBlock(decoder_dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(decoder_dim, elementwise_affine=False, eps=1e-6)
        self.norm_ada = nn.Sequential(nn.SiLU(), nn.Linear(decoder_dim, 2 * decoder_dim))
        nn.init.zeros_(self.norm_ada[-1].weight)
        nn.init.zeros_(self.norm_ada[-1].bias)

        self.pixel_head = nn.Linear(decoder_dim, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.pixel_head.weight)
        nn.init.zeros_(self.pixel_head.bias)

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, latent):
        if latent.dim() == 3:                 # [B, 1, D] -> [B, D]
            latent = latent.squeeze(1)
        B = latent.shape[0]

        c = self.cond(latent)                 # [B, decoder_dim]
        x = self.pos_embedding.expand(B, -1, -1)
        for block in self.blocks:
            x = block(x, c)

        shift, scale = self.norm_ada(c).chunk(2, dim=-1)
        x = _modulate(self.norm(x), shift, scale)
        patches = self.pixel_head(x)

        image = rearrange(
            patches, "b (h w) (p1 p2 ch) -> b ch (h p1) (w p2)",
            h=self.grid_size, w=self.grid_size,
            p1=self.patch_size, p2=self.patch_size, ch=self.out_channels,
        )
        image = image + self.refine(image)
        return torch.sigmoid(image)
