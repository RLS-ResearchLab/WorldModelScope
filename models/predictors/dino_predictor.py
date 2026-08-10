import torch
from torch import nn
from einops import rearrange


def generate_causal_mask(
    num_frames: int,
    num_tokens_per_frame: int,
    device=None,
):
    """
    Create a spatio-temporal causal attention mask.

    Each frame contributes `num_tokens_per_frame` tokens to the sequence.
    In DINO-WM this is (num_visual_patches + 1), where the extra token
    is the action embedding appended to that frame's visual tokens by
    DINOWM.predict(). This function itself doesn't care what the tokens
    represent — it only needs the per-frame token count to build blocks.

    A token from frame t can attend to:
        - all tokens from frames <= t

    It cannot attend to:
        - tokens from future frames > t

    Returns:
        mask: (1, 1, num_frames * num_tokens_per_frame,
                    num_frames * num_tokens_per_frame)
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
    # num_tokens_per_frame x num_tokens_per_frame block.
    #
    # Result:
    #
    # (num_frames * num_tokens_per_frame,
    #  num_frames * num_tokens_per_frame)
    mask = frame_mask.repeat_interleave(
        num_tokens_per_frame,
        dim=0,
    ).repeat_interleave(
        num_tokens_per_frame,
        dim=1,
    )

    # Add batch and attention-head dimensions.
    #
    # (T*N, T*N)
    #
    # -> (1, 1, T*N, T*N)
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
        N = num_frames * num_tokens_per_frame
        D = embedding dimension

    num_tokens_per_frame = num_visual_patches + 1 (action token),
    as assembled by DINOWM.predict() before this module ever sees the
    sequence. This module is agnostic to that fact — it just needs the
    count to build the correctly-shaped causal mask.
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
                num_tokens_per_frame=num_patches,
            ),
            persistent=False,
        )

    def forward(self, x):

        # x:
        #
        # (B, N, D)
        #
        # N = T * (P + 1)

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
        x: (B, T * N, D)

    where:
        B = batch size
        T = number of frames in the window (num_hist + num_pred)
        N = tokens per frame = num_visual_patches + 1
            (the "+1" is the action token DINOWM.predict() concatenates
            onto each frame's visual patch tokens before calling this
            module — see DINOWM.predict())
        D = shared embedding dimension (post feature_adapter, so it's
            identical regardless of which frozen encoder — DINOv2,
            DUNE, VGGT, ... — produced the visual tokens)

    Output:
        x: (B, T * N, D)

    The predictor operates entirely in latent space and is unaware of
    which token slots are "visual" vs. "action" — DINOWM is responsible
    for slicing action-token outputs back out after calling this module.

    Example:

        DINOv2 patch tokens per frame: 256 (P) -> 257 (P+1) after
        the action token is appended.

        emb_dim = 384, T = num_hist + num_pred = 4

        Before predictor:
            (B, 4 * 257, 384)
        After predictor:
            (B, 4 * 257, 384)
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

        # NOTE: `num_patches` here is P + 1 (visual patches + action
        # token). Callers (see models/factory.py::build_model) must
        # pass num_patches=P+1, not the raw encoder patch count.
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
        # (1, T*N, D)
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
        B, N, D = x.shape

        if D != self.dim:
            raise ValueError(
                f"ViTPredictor expected embedding dimension {self.dim}, "
                f"but received {D}."
            )

        max_tokens = self.num_frames * self.num_patches

        if N > max_tokens:
            raise ValueError(
                f"ViTPredictor received {N} tokens, which exceeds the "
                f"maximum window it was built for "
                f"({self.num_frames} frames x {self.num_patches} tokens/frame "
                f"= {max_tokens})."
            )

        if N % self.num_patches != 0:
            raise ValueError(
                f"ViTPredictor received {N} tokens, which is not a whole "
                f"number of frames (tokens/frame = {self.num_patches})."
            )

        x = x + self.pos_embedding[:, :N]
        x = self.dropout(x)
        x = self.transformer(x)

        return x