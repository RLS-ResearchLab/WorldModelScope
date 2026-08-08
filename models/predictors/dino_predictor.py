import torch
from torch import nn
from einops import rearrange


def generate_causal_mask(
    num_frames: int,
    num_patches: int,
    device=None,
):
    """
    Create a spatio-temporal causal attention mask.

    Each frame contains `num_patches` visual tokens.

    A token from frame t can attend to:
        - all tokens from frames <= t

    It cannot attend to:
        - tokens from future frames > t

    Returns:
        mask: (1, 1, num_frames * num_patches,
                    num_frames * num_patches)
    """

    # Frame-level causal mask.
    #
    # Example for 3 frames:
    #
    # [[1, 0, 0],
    #  [1, 1, 0],
    #  [1, 1, 1]]
    frame_mask = torch.tril(
        torch.ones(
            num_frames,
            num_frames,
            device=device,
        )
    )

    # Expand every frame-level entry into a
    # num_patches x num_patches block.
    #
    # Result:
    #
    # (num_frames * num_patches,
    #  num_frames * num_patches)
    mask = frame_mask.repeat_interleave(
        num_patches,
        dim=0,
    ).repeat_interleave(
        num_patches,
        dim=1,
    )

    # Add batch and attention-head dimensions.
    #
    # (T*P, T*P)
    #
    # -> (1, 1, T*P, T*P)
    return mask.unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    """
    Transformer feed-forward block.

    DINO-WM predictor uses a standard Transformer-style MLP
    after the attention block.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
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


class Attention(nn.Module):
    """
    Multi-head self-attention with temporal causal masking.

    Input:
        x: (B, N, D)

    where:
        N = num_frames * num_patches
        D = embedding dimension
    """

    def __init__(
        self,
        dim: int,
        num_frames: int,
        num_patches: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.heads = heads
        self.scale = dim_head ** -0.5

        inner_dim = heads * dim_head

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias=False,
        )

        self.attend = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

        self.num_frames = num_frames
        self.num_patches = num_patches

        # Register the mask as a buffer.
        #
        # when the model is moved to:
        #
        #     cuda
        #     cpu
        #     another device
        #
        # the mask moves with it.
        self.register_buffer(
            "causal_mask",
            generate_causal_mask(
                num_frames=num_frames,
                num_patches=num_patches,
            ),
            persistent=False,
        )

    def forward(self, x):

        # x:
        #
        # (B, N, D)
        #
        # N = T * P

        B, N, D = x.shape

        x = self.norm(x)

        # Create Q, K and V.
        #
        # Each:
        #
        # (B, N, heads * dim_head)

        q, k, v = self.to_qkv(x).chunk(
            3,
            dim=-1,
        )

        # Convert into multi-head representation:
        #
        # (B, N, H*Dh)
        #
        # ->
        #
        # (B, H, N, Dh)

        q = rearrange(
            q,
            "b n (h d) -> b h n d",
            h=self.heads,
        )

        k = rearrange(
            k,
            "b n (h d) -> b h n d",
            h=self.heads,
        )

        v = rearrange(
            v,
            "b n (h d) -> b h n d",
            h=self.heads,
        )

        # Scaled dot-product attention.
        #
        # (B, H, N, Dh)
        #
        # x
        #
        # (B, H, Dh, N)
        #
        # ->
        #
        # (B, H, N, N)

        dots = torch.matmul(
            q,
            k.transpose(-1, -2),
        ) * self.scale

        # Make sure the mask has the correct size.
        mask = self.causal_mask[
            ...,
            :N,
            :N,
        ]

        # Prevent attending to future frames.
        dots = dots.masked_fill(
            mask == 0,
            torch.finfo(dots.dtype).min,
        )

        # Convert attention scores into probabilities.
        attn = self.attend(dots)

        attn = self.dropout(attn)

        # Apply attention to V.
        #
        # (B, H, N, N)
        #
        # x
        #
        # (B, H, N, Dh)
        #
        # ->
        #
        # (B, H, N, Dh)

        out = torch.matmul(
            attn,
            v,
        )

        # Merge attention heads.
        #
        # (B, H, N, Dh)
        #
        # ->
        #
        # (B, N, H*Dh)

        out = rearrange(
            out,
            "b h n d -> b n (h d)",
        )

        # Final projection.
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer used as the DINO-WM latent predictor.
    """

    def __init__(
        self,
        dim: int,
        num_frames: int,
        num_patches: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(depth):

            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            num_frames=num_frames,
                            num_patches=num_patches,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(
                            dim=dim,
                            hidden_dim=mlp_dim,
                            dropout=dropout,
                        ),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        for attention, feed_forward in self.layers:

            # Attention + residual connection.
            x = attention(x) + x

            # Feed-forward + residual connection.
            x = feed_forward(x) + x

        return self.norm(x)


class ViTPredictor(nn.Module):
    """
    ViT-based latent predictor used in DINO-WM.

    Input:
        x: (B, T * P, D)

    where:
        B = batch size
        T = number of frames
        P = number of patches/tokens per frame
        D = encoder embedding dimension

    Output:
        x: (B, T * P, D)

    The predictor operates entirely in latent space.

    Example:

        DINOv2 patch tokens:
            (B, T, 256, 384)

        Before predictor:
            (B, T * 256, 384)

        After predictor:
            (B, T * 256, 384)
    """

    def __init__(
        self,
        *,
        num_patches: int,
        num_frames: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.num_frames = num_frames
        self.dim = dim

        # Total number of tokens processed by the predictor.
        self.num_tokens = (
            num_frames * num_patches
        )

        # Learned spatio-temporal positional embeddings.
        #
        # Shape:
        #
        # (1, T*P, D)
        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_tokens,
                dim,
            )
        )

        self.dropout = nn.Dropout(
            emb_dropout
        )

        self.transformer = Transformer(
            dim=dim,
            num_frames=num_frames,
            num_patches=num_patches,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x):
        """
        Args:
            x:
                (B, T*P, D)

        Returns:
            (B, T*P, D)
        """

        B, N, D = x.shape

        expected_tokens = (
            self.num_frames *
            self.num_patches
        )

        # Make sure the input matches the predictor
        # configuration.
        if N != expected_tokens:
            raise ValueError(
                f"ViTPredictor expected "
                f"{expected_tokens} tokens "
                f"({self.num_frames} frames x "
                f"{self.num_patches} patches), "
                f"but received {N}."
            )

        if D != self.dim:
            raise ValueError(
                f"ViTPredictor expected embedding "
                f"dimension {self.dim}, "
                f"but received {D}."
            )

        # Add learned positional embeddings.
        x = (
            x
            + self.pos_embedding[:, :N]
        )

        # Embedding dropout.
        x = self.dropout(x)

        # Transformer predictor.
        x = self.transformer(x)

        return x