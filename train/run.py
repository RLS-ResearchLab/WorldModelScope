import argparse
from src.utils.config import load_config

from models.factory import build_model

from data.dataloader import build_dataloader

from src.utils.optimizer import build_optimizers
from src.utils.scheduler import build_schedulers
from src.utils.logger import build_logger
from src.utils.checkpoints import CheckpointManager

from train.run import Trainer


def main(config_path):

    config = load_config(config_path)

    model = build_model(config)

    train_loader = build_dataloader(config)

    optimizers = build_optimizers(
        model,
        config["optimizer"],
    )

    schedulers = build_schedulers(
        optimizers,
        config["scheduler"],
    )

    logger = build_logger(
        config["logging"],
    )

    checkpoint = CheckpointManager(
        config["checkpoint"],
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

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    main(args.config)
