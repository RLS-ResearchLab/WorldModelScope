import torch
import torch.nn as nn


class DINOWMActionEncoder(nn.Module):
    """
    Action encoder used by the original DINO-WM implementation.

    The official DINO-WM implementation uses a Conv1d projection
    to map action vectors into the predictor embedding dimension.

    Input:
        x:
            (B, T, action_dim)

    Output:
        action_embedding:
            (B, T, emb_dim)

    Example:

        action_dim = 7
        emb_dim = 384

        (B, 3, 7)
            |
            v
        Conv1d(7, 384, kernel_size=1)
            |
            v
        (B, 3, 384)
    """

    def __init__(
        self,
        action_dim: int,
        emb_dim: int,
        tubelet_size: int = 1,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.tubelet_size = tubelet_size

        if tubelet_size != 1:
            raise NotImplementedError(
                "DINO-WM currently uses "
                "tubelet_size=1 for actions."
            )

        self.patch_embed = nn.Conv1d(
            in_channels=action_dim,
            out_channels=emb_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size,
        )

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(
                "Expected actions with shape "
                f"(B, T, action_dim), got {x.shape}"
            )

        B, T, D = x.shape
          
        if D != self.action_dim:
            raise ValueError(
                f"Expected action dimension "
                f"{self.action_dim}, got {D}"
            )

        # Conv1d expects:
        #
        # (B, channels, sequence_length)
        #
        # Current:
        #
        # (B, T, action_dim)
        #
        # -> (B, action_dim, T)

        x = x.permute(0, 2, 1)

        x = self.patch_embed(x)

        # Back to:
        #
        # (B, T, emb_dim)

        x = x.permute(0, 2, 1)

        return x
