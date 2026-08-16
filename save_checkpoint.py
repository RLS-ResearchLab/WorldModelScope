from pathlib import Path
import wandb

run = wandb.init(project="dino-wm-bridge", job_type="upload-checkpoint")

ckpt_dir = Path("/path/to/your/old/checkpoints")  # <-- put your real local folder here

for ckpt_file in ckpt_dir.glob("*.pt"):
    artifact = wandb.Artifact(name=ckpt_file.stem, type="model")
    artifact.add_file(str(ckpt_file))
    run.log_artifact(artifact)

run.finish()