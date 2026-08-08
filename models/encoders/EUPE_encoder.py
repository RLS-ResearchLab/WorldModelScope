
"""
EUPE-ViT-S encoder wrapper for the World Model.

Input:
    images: [B, 3, 224, 224]

Output:
    patch tokens: [B, 196, 384]

The EUPE encoder is frozen.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# Locate the official EUPE repository
# ============================================================

EUPE_ROOT = Path(r"C:\world_model\EUPE")

if str(EUPE_ROOT) not in sys.path:
    sys.path.insert(0, str(EUPE_ROOT))


from eupe.hub.backbones import eupe_vits16


class EUPEEncoder(nn.Module):
    """
    Frozen EUPE-ViT-S encoder.

    EUPE-ViT-S configuration:

        Image size     : 224 x 224
        Patch size     : 16 x 16
        Number patches : 14 x 14 = 196
        Embedding dim  : 384

    Output:
        [B, 196, 384]
    """

    def __init__(
        self,
        checkpoint_path: str,
        freeze: bool = True,
    ):
        super().__init__()

        self.embed_dim = 384
        self.num_patches = 196
        self.patch_size = 16

        # ----------------------------------------------------
        # Build the official EUPE architecture
        # ----------------------------------------------------

        self.model = eupe_vits16(
            pretrained=False
        )

        # ----------------------------------------------------
        # Load pretrained EUPE checkpoint
        # ----------------------------------------------------

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        missing_keys, unexpected_keys = (
            self.model.load_state_dict(
                checkpoint,
                strict=False
            )
        )

        if missing_keys:
            print(
                "EUPE missing keys:",
                missing_keys
            )

        if unexpected_keys:
            print(
                "EUPE unexpected keys:",
                unexpected_keys
            )

        # ----------------------------------------------------
        # Freeze EUPE
        # ----------------------------------------------------

        if freeze:

            for parameter in self.model.parameters():
                parameter.requires_grad = False

            self.model.eval()

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, images):
        """
        Args:
            images:
                [B, 3, 224, 224]

        Returns:
            patch_tokens:
                [B, 196, 384]
        """

        # ----------------------------------------------------
        # EUPE forward
        # ----------------------------------------------------

        features = self.model.forward_features(images)

        # ----------------------------------------------------
        # Keep ONLY spatial patch tokens.
        #
        # EUPE internally produces:
        #
        #   1 CLS token
        #   4 storage tokens
        #   196 patch tokens
        #
        # We only need the 196 spatial tokens.
        # ----------------------------------------------------

        patch_tokens = features[
            "x_norm_patchtokens"
        ]

        return patch_tokens

