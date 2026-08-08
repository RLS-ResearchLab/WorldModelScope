
import torch

from models.encoders.EUPE_encoder import EUPEEncoder


def main():

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        r"C:\Users\lenovo\.cache\huggingface\hub"
        r"\models--facebook--EUPE-ViT-S"
        r"\snapshots\b741b808d19487475c225a7e672e4842f32cd402"
        r"\EUPE-ViT-S.pt"
    )


    # --------------------------------------------------------
    # Create OUR encoder wrapper
    # --------------------------------------------------------

    encoder = EUPEEncoder(
        checkpoint_path=checkpoint_path,
        freeze=True,
    ).to(device)


    # --------------------------------------------------------
    # Dummy images
    # --------------------------------------------------------

    images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    with torch.no_grad():

        features = encoder(images)


    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print("\nInput:")
    print(images.shape)

    print("\nEncoder output:")
    print(features.shape)

    print("\nEmbedding dimension:")
    print(encoder.embed_dim)

    print("\nNumber of patches:")
    print(encoder.num_patches)


    # --------------------------------------------------------
    # Assertions
    # --------------------------------------------------------

    assert features.shape == (
        2,
        196,
        384,
    )

    trainable = sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )

    assert trainable == 0

    print("\nSUCCESS")
    print("EUPEEncoder is ready for the World Model.")


if __name__ == "__main__":
    main()
