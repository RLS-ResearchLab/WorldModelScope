import wandb

api = wandb.Api()

artifact_names = [
    "dino_wm-epoch_0048:v0",
    "dino_wm-epoch_0049:v0",
    "dino_wm-epoch_0050:v0",
    "dino_wm-latest:v0",
    "eupe_encoder-epoch_0048:v0",
    "eupe_encoder-epoch_0049:v0",
    "eupe_encoder-epoch_0050:v0",
    "eupe_encoder-latest:v0",
]

for name in artifact_names:
    artifact = api.artifact(f"mahfoudhsamar-sup-com/dino-wm-bridge/{name}")

    encoder = name.split("-")[0]  # "dino_wm" or "eupe_encoder"
    target_dir = f"checkpoints/{encoder}"

    artifact.download(root=target_dir)
    print(f"Downloaded {name} -> {target_dir}")