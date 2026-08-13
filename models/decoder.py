import torch
import torch.nn as nn
import torch.nn.functional as F

# ImageNet stats DINOv2 was pretrained with. Applied here so the decoder
# is trained on the SAME token distribution your DINO-WM predictor sees.
# If your DINO-WM pipeline feeds raw [0,1] images to this encoder WITHOUT
# normalization, remove this and pass images through unnormalized instead
# -- the two must match.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOv2Encoder(nn.Module):
    """
    Frozen pretrained DINOv2 encoder.

    Input:
        images: [B, 3, H, W], values in [0, 1]

    Output:
        patch_tokens: [B, N, D]
    """

    def __init__(
        self,
        model_name="dinov2_vits14",
        freeze=True,
        normalize=True,
    ):
        super().__init__()

        # Load pretrained DINOv2
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
        )

        self.embed_dim = self.model.embed_dim
        self.normalize = normalize

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
            images: [B, 3, H, W], values in [0, 1]

        Returns:
            patch_tokens: [B, N, D]
        """
        images = F.interpolate(
            images,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        if self.normalize:
            mean = IMAGENET_MEAN.to(images.device)
            std = IMAGENET_STD.to(images.device)
            images = (images - mean) / std

        features = self.model.forward_features(images)

        patch_tokens = features["x_norm_patchtokens"]

        return patch_tokens