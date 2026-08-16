from pathlib import Path
import wandb

run = wandb.init(project="dino-wm-bridge", job_type="upload-checkpoint")

base_dir = Path("/home/riftuser/WorldModelScope/checkpoints")

for subfolder in base_dir.iterdir():
    if not subfolder.is_dir():
        continue

    for ckpt_file in subfolder.glob("*.pt"):
        # artifact name includes the encoder folder, e.g. "eupe_encoder-epoch_0050"
        artifact_name = f"{subfolder.name}-{ckpt_file.stem}"

        artifact = wandb.Artifact(name=artifact_name, type="model")
        artifact.add_file(str(ckpt_file))
        run.log_artifact(artifact)

        print(f"Uploaded: {ckpt_file} -> {artifact_name}")

run.finish()