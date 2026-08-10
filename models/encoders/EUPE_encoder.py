"""
EUPE ViT-Small encoder.

Loads the pretrained EUPE ViT-Small model through timm.
Weights are automatically downloaded from Hugging Face on first use
and cached locally for subsequent runs.

Input:
    [B, 3, H, W]

Output:
    [B, N, 384]
    where N = (H / 16) * (W / 16)

Example:
    224x224 -> [B, 196, 384]
    256x256 -> [B, 256, 384]
"""

import torch
import torch.nn as nn
import timm


class EUPEEncoder(nn.Module):
    """
    Frozen EUPE ViT-Small encoder.

    Uses:
        vit_small_patch16_dinov3_qkvb.eupe_lvd1689m

    Output:
        Patch-level EUPE representations.
    """

    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "vit_small_patch16_dinov3_qkvb.eupe_lvd1689m",
            pretrained=True,
            num_classes=0,
        )

        self.model.eval()

        # EUPE ViT-Small
        self.embed_dim = 384
        self.patch_size = 16

        # CLS + register/storage tokens
        self.num_prefix_tokens = self.model.num_prefix_tokens

        # Freeze encoder
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]

        Returns:
            Patch tokens: [B, N, 384]
        """

        features = self.model.forward_features(x)

        # Remove CLS/register tokens.
        patch_tokens = features[:, self.num_prefix_tokens:, :]

        return patch_tokens