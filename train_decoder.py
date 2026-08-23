import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.utils import make_grid, save_image
import wandb

from train import Trainer
from src.utils.config import load_config
from src.utils.checkpoints import find_latest_checkpoint, peek_wandb_run_id
from src.utils.optimizer import build_optimizer
from src.utils.scheduler import build_scheduler
from src.utils.logger import Logger, WandbLogger, CombinedLogger

from datasets.dataloader import build_dataloaders
from models.decoder.decoder_builder import EncoderDecoderModel


class DecoderTrainer(Trainer):
    def __init__(self, *args, wandb_logger=None, num_visualize=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb_logger = wandb_logger
        self.num_visualize = num_visualize
        self.image_dir = Path(self.config["logging"]["output_dir"]) / "images"

    def prepare_batch(self, raw_batch):
        return {"observations": raw_batch["frames"]}

    def validate(self):
        val_metrics = super().validate()
        self._log_visualizations()
        return val_metrics

    @torch.no_grad()
    def _log_visualizations(self):
        raw_batch = self._move_batch(next(iter(self.val_loader)))
        batch = self.prepare_batch(raw_batch)
        vis = self.model.get_visualizations(batch, max_images=self.num_visualize)

        # top row: ground truth, bottom row: reconstruction, same column = same frame
        grid = make_grid(
            torch.cat([vis["ground_truth"], vis["reconstruction"]], dim=0),
            nrow=vis["ground_truth"].shape[0],
        )

        self.image_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, self.image_dir / f"step_{self.global_step:07d}.png")

        if self.wandb_logger is not None:
            self.wandb_logger.run.log(
                {"val/reconstructions": wandb.Image(grid)}, step=self.global_step,
            )


def main(config_path):
    config = load_config(config_path)

    resume_path = config.get("resume_from")
    if resume_path == "auto":
        resume_path = find_latest_checkpoint(config["checkpoint"]["dir"])
    resume_wandb_id = peek_wandb_run_id(resume_path) if resume_path else None

    model = EncoderDecoderModel(config)

    # build_dataloaders/build_dataloader expect a flat config["model"]["image_size"] (the
    # dino_wm config shape) -- ours nests the model block under config["dino_wm"]["model"],
    # so adapt a view rather than changing the shared dataloader code.
    loader_cfg = OmegaConf.create({"data": config["data"], "model": config["dino_wm"]["model"]})
    loaders = build_dataloaders(loader_cfg)

    optimizer = build_optimizer(model, config["optimizer"])
    scheduler = build_scheduler(optimizer, config["scheduler"], config["training"]["max_steps"])

    local_logger = Logger(config["logging"]["output_dir"])
    wandb_logger = WandbLogger(
        project="dino-wm-decoder", name=config.get("run_name"),
        output_dir=config["logging"]["output_dir"], config=config, resume_id=resume_wandb_id,
    )
    logger = CombinedLogger([local_logger, wandb_logger])

    trainer = DecoderTrainer(
        model=model, train_loader=loaders["train"], val_loader=loaders["val"],
        optimizers={"main": optimizer}, schedulers={"main": scheduler},
        logger=logger, checkpoint_dir=config["checkpoint"]["dir"], config=config,
        wandb_logger=wandb_logger,
    )
    trainer.wandb_run_id = logger.get_wandb_run_id()

    if resume_path:
        trainer.load_checkpoint(resume_path)

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    main(parser.parse_args().config)
