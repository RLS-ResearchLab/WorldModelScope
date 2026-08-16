
import argparse
from pathlib import Path

import torch

from train import Trainer
from src.utils.config import load_config
from src.utils.checkpoints import CheckpointManager
from src.utils.optimizer import build_optimizer
from src.utils.scheduler import build_scheduler
from src.utils.logger import Logger, WandbLogger, CombinedLogger

from datasets.dataloader import build_dataloaders

from models.decoder.decoder_model import DecoderModel



class DecoderTrainer(Trainer):

    def prepare_batch(self, raw_batch):
        return {"observations": raw_batch["frames"]}

def build_logger(config, output_dir):
    loggers = [Logger(output_dir)]

    if config["wandb"]["enabled"]:
        loggers.append(
            WandbLogger(
                project=config["wandb"]["project"],
                name=config["wandb"]["name"],
                config=dict(config),
            )
        )

    return CombinedLogger(loggers)

def main(config_path):
    config = load_config(config_path)

    device = torch.device(
        config["training"].get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(config, output_dir)
    logger.save_config(dict(config))

    loaders = build_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    model = DecoderModel(config).to(device)

    optimizer = build_optimizer(model, config["optimizer"])

    
    steps_per_epoch = config["training"].get("steps_per_epoch")
    if not steps_per_epoch:
        raise ValueError(
            "config['training']['steps_per_epoch'] is required: "
            "BridgeDataset is an IterableDataset and has no len(), "
            "so it can't be inferred from the loader."
        )
    total_steps = steps_per_epoch * config["training"]["epochs"]
    scheduler = build_scheduler(optimizer, config["scheduler"], total_steps)

    checkpoint_manager = CheckpointManager(
        output_dir / "checkpoints",
        keep_last=config["training"].get("keep_last_checkpoints", 3),
    )

    trainer = DecoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers={"decoder": optimizer},
        schedulers={"decoder": scheduler},
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        config=config,
    )

    resume_from = config["training"].get("resume_from")
    if resume_from:
        trainer.load_checkpoint(resume_from)

    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)