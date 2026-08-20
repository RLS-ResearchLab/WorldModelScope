import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _HAS_SDPA_KERNEL_CTX = True
except ImportError:
    _HAS_SDPA_KERNEL_CTX = False


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

    mask = frame_mask.repeat_interleave(
        num_tokens_per_frame,
        dim=0,
    ).repeat_interleave(
        num_tokens_per_frame,
        dim=1,
    )

    return mask.unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):

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

    def __init__(
        self,
        dim: int,
        num_frames: int,
        num_patches: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        sdpa_backend: str = "auto",
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
        # "auto" leaves backend selection to PyTorch (may fall back to the
        # memory-heavy math kernel with a custom mask). "efficient" forces
        # the memory-efficient backend, which does NOT materialize the full
        # N x N score matrix and is what actually saves memory here.
        self.sdpa_backend = sdpa_backend

        self.register_buffer(
            "causal_mask",
            generate_causal_mask(
                num_frames=num_frames,
                num_tokens_per_frame=num_patches,
            ),
            persistent=False,
        )

    def forward(self, x):
        B, N, D = x.shape
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        mask = self.causal_mask[..., :N, :N].bool()  # SDPA wants bool: True = keep
        dropout_p = self.dropout.p if self.training else 0.0

        if self.sdpa_backend == "efficient" and _HAS_SDPA_KERNEL_CTX:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout_p,
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p,
            )

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Transformer used as the DINO-WM latent predictor.

    use_checkpoint=True wraps each attention/feedforward sublayer in
    torch.utils.checkpoint, so activations for that sublayer are freed
    after the forward pass and recomputed during backward instead of
    being kept resident for all `depth` layers simultaneously. This is
    the single biggest lever for fitting a longer num_hist in memory --
    it trades ~25-30% more compute time for roughly a 1/depth reduction
    in stored activation memory.
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
        use_checkpoint: bool = True,
        sdpa_backend: str = "auto",
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
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
                            sdpa_backend=sdpa_backend,
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

    def _attn_block(self, attention, x):
        return attention(x) + x

    def _ff_block(self, feed_forward, x):
        return feed_forward(x) + x

    def forward(self, x):

        do_checkpoint = self.use_checkpoint and self.training

        for attention, feed_forward in self.layers:

            if do_checkpoint:
                x = checkpoint(
                    self._attn_block, attention, x, use_reentrant=False
                )
                x = checkpoint(
                    self._ff_block, feed_forward, x, use_reentrant=False
                )
            else:
                x = self._attn_block(attention, x)
                x = self._ff_block(feed_forward, x)

        return self.norm(x)


class ViTPredictor(nn.Module):

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
        use_checkpoint: bool = True,
        sdpa_backend: str = "auto",
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
            use_checkpoint=use_checkpoint,
            sdpa_backend=sdpa_backend,
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