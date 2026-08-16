
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DecoderFeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderAttention(nn.Module):

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        self.heads = heads
        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
        self.dropout = dropout

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class DecoderTransformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        DecoderAttention(dim, heads, dim_head, dropout),
                        DecoderFeedForward(dim, mlp_dim, dropout),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attention, feed_forward in self.layers:
            x = attention(x) + x
            x = feed_forward(x) + x

        return self.norm(x)


class ConvRefinement(nn.Module):

    def __init__(self, channels=3, hidden=64, num_blocks=3):
        super().__init__()

        layers = [
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        ]

        for _ in range(num_blocks):
            layers += [
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GroupNorm(8, hidden),
                nn.GELU(),
            ]

        layers += [nn.Conv2d(hidden, channels, kernel_size=3, padding=1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)

class ViTDecoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_patches,
        img_size,
        patch_size,
        decoder_dim=384,
        num_layers=6,
        num_heads=6,
        dim_head=64,
        mlp_ratio=4.0,
        out_channels=3,
        dropout=0.0,
        use_conv_refinement=True,
        refinement_hidden=64,
        refinement_blocks=3,
    ):
        super().__init__()

        grid_size = img_size // patch_size

        if grid_size * grid_size != num_patches:
            raise ValueError(
                f"num_patches ({num_patches}) does not match the "
                f"img_size/patch_size grid "
                f"({grid_size}x{grid_size}={grid_size * grid_size}). "
                f"num_patches must equal the number of tokens the "
                f"encoder actually outputs for this img_size."
            )

        self.num_patches = num_patches
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.out_channels = out_channels

        # Project encoder/predictor latent width to the decoder's own width.
        self.input_proj = nn.Linear(latent_dim, decoder_dim)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, decoder_dim) * 0.02
        )

        self.transformer = DecoderTransformer(
            dim=decoder_dim,
            depth=num_layers,
            heads=num_heads,
            dim_head=dim_head,
            mlp_dim=int(decoder_dim * mlp_ratio),
            dropout=dropout,
        )

        self.pixel_head = nn.Linear(
            decoder_dim,
            patch_size * patch_size * out_channels,
        )

        self.use_conv_refinement = use_conv_refinement

        if use_conv_refinement:
            self.refine = ConvRefinement(
                channels=out_channels,
                hidden=refinement_hidden,
                num_blocks=refinement_blocks,
            )

    def forward(self, latents):
        B, N, D = latents.shape

        if N != self.num_patches:
            raise ValueError(
                f"ViTDecoder was built for {self.num_patches} tokens "
                f"but received {N}."
            )

        x = self.input_proj(latents) + self.pos_embedding
        x = self.transformer(x)

        patches = self.pixel_head(x)  # [B, N, p*p*C]

        image = rearrange(
            patches,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=self.grid_size,
            w=self.grid_size,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.out_channels,
        )

        if self.use_conv_refinement:
            image = self.refine(image)

        return image