""" Input image:        224 × 224
Encoder:            DINOv2 ViT-S/14
Patch size:         14 × 14
Feature:            x_norm_patchtokens
Feature dimension: 384
Number of patches:  16 × 16 = 256
Output:             [B, 256, 384]
Encoder:            frozen"""


import torch

from models.encoders.dinov2 import DINOv2Encoder


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    encoder = DINOv2Encoder(
        model_name="dinov2_vits14",
        freeze=True,
    ).to(device)

    images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    with torch.no_grad():
        features = encoder(images)

    print("Input shape :", images.shape)
    print("Output shape:", features.shape)
    print("Embedding dim:", encoder.embed_dim)

    # Check that encoder is frozen
    trainable = sum(
        p.requires_grad
        for p in encoder.parameters()
    )

    print("Trainable parameters:", trainable)


if __name__ == "__main__":
    main()