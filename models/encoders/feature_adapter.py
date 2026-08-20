import math

import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    Projects raw encoder features (native_dim) into the predictor's
    embedding dimension (emb_dim), with an optional spatial pooling
    step that reduces the number of patch tokens P before projection.

    Why pool: predictor sequence length is N = num_hist * (P + 1), and
    attention cost scales with N^2. P is usually the largest single
    contributor to N (e.g. ~256-370 tokens for a 224-256px DINOv2 grid),
    so shrinking P is often more effective than shrinking num_hist.

    pool_size=1 (default) disables pooling and preserves the original
    behavior -- a plain linear projection from native_dim to emb_dim.
    pool_size=2 averages each 2x2 block of the patch grid, cutting P
    to roughly P/4 (e.g. 256 -> 64 patches).
    """

    def __init__(self, native_dim: int, emb_dim: int, pool_size: int = 1):
        super().__init__()

        if pool_size < 1:
            raise ValueError("pool_size must be >= 1")

        self.pool_size = pool_size
        self.proj = nn.Linear(native_dim, emb_dim)

    def _pool_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """
        features: (B, T, P, D) with P assumed to be a perfect square
        (a square patch grid, as produced by ViT-style encoders).
        Returns: (B, T, P // pool_size**2, D)
        """
        B, T, P, D = features.shape
        side = int(math.isqrt(P))

        if side * side != P:
            raise ValueError(
                f"FeatureAdapter pooling expects a square patch grid, "
                f"but got P={P} (not a perfect square). Set pool_size=1 "
                f"to disable pooling, or adjust the encoder's patch count."
            )

        if side % self.pool_size != 0:
            raise ValueError(
                f"Patch grid side {side} is not divisible by "
                f"pool_size {self.pool_size}."
            )

        x = features.reshape(B * T, side, side, D)
        x = x.permute(0, 3, 1, 2)  # (B*T, D, side, side)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=self.pool_size)
        new_side = side // self.pool_size
        x = x.permute(0, 2, 3, 1).reshape(B, T, new_side * new_side, D)
        return x

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, P, D_native)
        if self.pool_size > 1:
            features = self._pool_tokens(features)
        return self.proj(features)