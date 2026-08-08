import torch
import torch.nn as nn


class DINOv2Encoder(nn.Module):
    """
    Frozen pretrained DINOv2 encoder.

    Input:
        images: [B, 3, H, W]

    Output:
        patch_tokens: [B, N, D]
    """

    def __init__(
        self,
        model_name="dinov2_vits14",
        freeze=True,
    ):
        super().__init__()

        # Load pretrained DINOv2
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
        )

        self.embed_dim = self.model.embed_dim

        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all DINOv2 parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    @torch.no_grad()
    def forward(self, images):
        """
        Extract DINOv2 spatial patch tokens.

        Args:
            images: [B, 3, H, W]

        Returns:
            patch_tokens: [B, N, D]
        """

        features = self.model.forward_features(images)

        patch_tokens = features["x_norm_patchtokens"]

        return patch_tokens