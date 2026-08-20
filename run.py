import argparse

from src.utils.config import load_config

from models.world_models.factory import build_model

from datasets.dataloader import build_dataloaders

from pathlib import Path

from src.utils.checkpoints import (
    find_latest_checkpoint,
    peek_wandb_run_id,
)

from src.utils.optimizer import build_optimizer
from src.utils.scheduler import build_scheduler
from src.utils.logger import Logger, WandbLogger, CombinedLogger


from train import Trainer


def main(config_path):

    config = load_config(config_path)

    resume_path = config.get("resume_from")

    if resume_path == "auto":
        resume_path = find_latest_checkpoint(
            config["checkpoint"]["dir"]
        )

    if resume_path is not None:
        resume_path = Path(resume_path)

        print(f"Resuming from: {resume_path}")

        resume_wandb_id = peek_wandb_run_id(resume_path)

    else:
        print("Starting a fresh training run.")
        resume_wandb_id = None

    model = build_model(config)

    loaders = build_dataloaders(config)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    optimizer = build_optimizer(
        model,
        config["optimizer"],
    )
    optimizers = {
        "main": optimizer,
    }

    scheduler = build_scheduler(
        optimizer,
        config["scheduler"],
        config["training"]["max_steps"]
    )
    schedulers = {
        "main": scheduler,
    }

    

    logger = CombinedLogger([
        Logger(config["logging"]["output_dir"]),
        WandbLogger(
            project="dino-wm",
            name=config.get("run_name"),
            output_dir=config["logging"]["output_dir"],
            config=config,
            resume_id=resume_wandb_id,
            
        ),
    ])

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        logger=logger,
        checkpoint_dir=config["checkpoint"]["dir"],
        config=config,
    )

    # Hand the (possibly new) wandb run id to the trainer so every checkpoint from here on
    # carries it forward -- covers both the fresh-run case and the resumed case.
    trainer.wandb_run_id = logger.get_wandb_run_id()

    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    main(args.config)