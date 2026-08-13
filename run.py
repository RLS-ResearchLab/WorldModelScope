import argparse
from src.utils.config import load_config

from models.world_models.factory import build_model

from datasets.dataloader import build_dataloader

from src.utils.optimizer import build_optimizer
from src.utils.scheduler import build_scheduler
from src.utils.logger import Logger, WandbLogger, CombinedLogger
from src.utils.checkpoints import CheckpointManager

from train import Trainer


def main(config_path):

    config = load_config(config_path)

    model = build_model(config)

    train_loader = build_dataloader(config)

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
        config["training"]["epochs"] * config["training"]["steps_per_epoch"]
    )
    schedulers = {
    "main": scheduler,
    }

    logger = CombinedLogger([
        Logger(config["logging"]["output_dir"]),
        WandbLogger(
            project="dino-wm-bridge",
            name=None,
            config=config,
        ),
    ])

    checkpoint = CheckpointManager(
        config["checkpoint"]["dir"],
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        logger=logger,
        checkpoint_manager=checkpoint,
        config=config,
    )

    trainer.fit()


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
