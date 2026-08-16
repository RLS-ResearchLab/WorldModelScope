import torch

from dataset import BridgeDataset


def main():

    dataset = BridgeDataset(
        data_dir="raw/bridge_dataset/1.0.0",
        split="train",
        clip_len=4,
        stride=8,
        camera_key="image_0",
        source_fps=5.0,
        target_fps=4.0,
        shuffle=False,
    )

    sample = next(iter(dataset))

    print("\n========== BRIDGE DATASET TEST ==========")

    print("frames:")
    print("  shape:", sample["frames"].shape)
    print("  dtype:", sample["frames"].dtype)
    print("  min:", sample["frames"].min().item())
    print("  max:", sample["frames"].max().item())

    print("\nactions:")
    print("  shape:", sample["actions"].shape)
    print("  dtype:", sample["actions"].dtype)

    print("\nstates:")
    print("  shape:", sample["states"].shape)
    print("  dtype:", sample["states"].dtype)

    print("\n==========================================")

    assert sample["frames"].ndim == 4
    assert sample["frames"].shape[0] == 4
    assert sample["frames"].shape[1] == 3

    assert sample["actions"].shape == (3, 7)

    assert sample["states"].shape[0] == 4
    assert sample["states"].shape[1] == 7

    print("\n✓ Dataset smoke test PASSED")


if __name__ == "__main__":
    main()