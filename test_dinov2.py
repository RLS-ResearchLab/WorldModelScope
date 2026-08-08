import torch


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # ---------------------------------------------------------
    # 1. Load official pretrained DINOv2 ViT-S/14
    # ---------------------------------------------------------
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
    )

    model = model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # 2. Freeze encoder
    # ---------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------
    # 3. Create test image
    # ---------------------------------------------------------
    images = torch.randn(
        2,
        3,
        224,
        224,
        device=device,
    )

    # ---------------------------------------------------------
    # 4. Forward pass
    # ---------------------------------------------------------
    with torch.no_grad():
        output = model.forward_features(images)

    # ---------------------------------------------------------
    # 5. Inspect official output
    # ---------------------------------------------------------
    print("\n===== DINOv2 =====")

    print("Input shape:")
    print(images.shape)

    print("\nOutput keys:")
    print(output.keys())

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:25s}: {value.shape}")

    # ---------------------------------------------------------
    # 6. Get spatial patch tokens
    # ---------------------------------------------------------
    patch_tokens = output["x_norm_patchtokens"]

    print("\nSpatial patch tokens:")
    print("Shape:", patch_tokens.shape)

    B, N, D = patch_tokens.shape

    print("\nExtracted dimensions:")
    print("B =", B)
    print("N =", N)
    print("D =", D)

    # ---------------------------------------------------------
    # 7. Frozen check
    # ---------------------------------------------------------
    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTrainable parameters:", trainable_params)

    assert trainable_params == 0


if __name__ == "__main__":
    main()