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

    @torch.no_grad()
    def validate(self):
        # Can't reuse Trainer.validate(): it windows each val clip into num_hist+1-length
        # slices (self.model.num_hist), which only makes sense for dino_wm's sequence
        # predictor. The decoder treats every frame independently, so each val batch is
        # scored as-is via prepare_batch, same as training.
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        metrics_sum = {}

        for raw_batch in self.val_loader:
            raw_batch = self._move_batch(raw_batch)
            batch = self.prepare_batch(raw_batch)

            with torch.autocast(
                device_type=self.device_type, dtype=self.amp_dtype, enabled=self.use_amp,
            ):
                output = self.model.validation_step(batch)
                loss, metrics = output if isinstance(output, tuple) else (output, {})

            total_loss += loss.detach().item()
            num_batches += 1
            for name, value in metrics.items():
                if torch.is_tensor(value):
                    value = value.detach().float().mean().item()
                metrics_sum[name] = metrics_sum.get(name, 0.0) + value

        self.model.train()

        val_metrics = {"val/loss": total_loss / max(num_batches, 1)}
        for name, value in metrics_sum.items():
            val_metrics[f"val/{name}"] = value / max(num_batches, 1)

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
