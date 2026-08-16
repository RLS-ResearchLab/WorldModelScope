import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from models.encoders.dinov2 import DINOv2Encoder
from models.decoder import DINODecoder

from datasets._temp.dataset import BridgeDataset

from src.utils.config import load_config
from src.utils.checkpoints import CheckpointManager
from src.utils.optimizer import build_optimizer
from src.utils.scheduler import build_scheduler
from src.utils.logger import (
    Logger,
    WandbLogger,
    CombinedLogger,
)



def build_decoder_dataloader(
    config,
    split,
):

    data_cfg = config["data"]

    params = data_cfg["params"]

    dataset = BridgeDataset(
        data_dir=params["data_dir"],
        split=split,
        clip_len=params["clip_len"],
        stride=params["stride"],
        camera_key=params["camera_key"],
        source_fps=params["source_fps"],
        target_fps=params["target_fps"],
        shuffle=(split == "train"),
    )

    loader = DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
    )

    return loader


# ============================================================
# Models
# ============================================================

def build_encoder(
    config,
    device,
):

    encoder = DINOv2Encoder(
        model_name=config["encoder"]["name"],
        freeze=True,
    )

    encoder = encoder.to(device)

    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    return encoder


def build_decoder(
    config,
    device,
):

    model_cfg = config["model"]

    decoder = DINODecoder(
        latent_dim=model_cfg["latent_dim"],
        num_patches=model_cfg["num_patches"],
        img_size=model_cfg["img_size"],
        patch_size=model_cfg["patch_size"],
        decoder_dim=model_cfg["decoder_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
    )

    return decoder.to(device)


# ============================================================
# One training epoch
# ============================================================

def train_one_epoch(
    encoder,
    decoder,
    loader,
    optimizer,
    scheduler,
    logger,
    device,
    epoch,
    global_step,
    config,
):

    decoder.train()

    image_size = config["model"]["img_size"]

    log_every = config["training"]["log_every"]

    running_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}",
    )

    for batch in progress:

        # ----------------------------------------------------
        # Bridge frames
        #
        # [B,T,C,H,W]
        # ----------------------------------------------------

        frames = batch["frames"].to(
            device,
            non_blocking=True,
        )

        B, T, C, H, W = frames.shape

        # ----------------------------------------------------
        # Flatten temporal dimension
        #
        # [B,T,C,H,W]
        #       ↓
        # [B*T,C,H,W]
        # ----------------------------------------------------

        frames = frames.reshape(
            B * T,
            C,
            H,
            W,
        )

        # ----------------------------------------------------
        # Resize exactly as DINO encoder does
        # ----------------------------------------------------

        frames = F.interpolate(
            frames,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

        # ----------------------------------------------------
        # Frozen DINOv2
        # ----------------------------------------------------

        with torch.no_grad():

            latents = encoder(
                frames
            )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        reconstruction = decoder(
            latents
        )

        # ----------------------------------------------------
        # Reconstruction loss
        # ----------------------------------------------------

        mse_loss = F.mse_loss(
            reconstruction,
            frames,
        )

        l1_loss = F.l1_loss(
            reconstruction,
            frames,
        )

        loss = (
            mse_loss
            + 0.1 * l1_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Scheduler
        #
        # IMPORTANT:
        # This is intentionally here.
        #
        # We do NOT modify src/utils/scheduler.py.
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Step
        # ----------------------------------------------------

        global_step += 1

        running_loss += loss.item()

        progress.set_postfix(
            loss=f"{loss.item():.6f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        # ----------------------------------------------------
        # Numeric logging
        # ----------------------------------------------------

        if global_step % log_every == 0:

            logger.log(
                global_step,
                {
                    "train/loss": loss.item(),
                    "train/mse": mse_loss.item(),
                    "train/l1": l1_loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
            )

    epoch_loss = (
        running_loss
        / max(len(loader), 1)
    )

    return epoch_loss, global_step


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    encoder,
    decoder,
    loader,
    device,
    image_size,
):

    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    count = 0

    visualization = None

    for batch in tqdm(
        loader,
        desc="Validation",
    ):

        frames = batch["frames"].to(
            device,
            non_blocking=True,
        )

        B, T, C, H, W = frames.shape

        frames = frames.reshape(
            B * T,
            C,
            H,
            W,
        )

        frames = F.interpolate(
            frames,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

        latents = encoder(
            frames
        )

        reconstruction = decoder(
            latents
        )

        mse_loss = F.mse_loss(
            reconstruction,
            frames,
        )

        l1_loss = F.l1_loss(
            reconstruction,
            frames,
        )

        loss = (
            mse_loss
            + 0.1 * l1_loss
        )

        total_loss += loss.item()

        count += 1

        # Keep one example
        if visualization is None:

            visualization = {
                "input": frames[0].detach().cpu(),
                "reconstruction": (
                    reconstruction[0]
                    .detach()
                    .cpu()
                ),
            }

    val_loss = (
        total_loss
        / max(count, 1)
    )

    return val_loss, visualization


# ============================================================
# Main
# ============================================================

def main(config_path):

    config = load_config(
        config_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = Path(
        config["experiment"]["output_dir"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Logger
    # ========================================================

    loggers = [
        Logger(output_dir)
    ]

    if config["wandb"]["enabled"]:

        loggers.append(
            WandbLogger(
                project=config["wandb"]["project"],
                name=config["wandb"]["name"],
                config=dict(config),
            )
        )

    logger = CombinedLogger(
        loggers
    )

    logger.save_config(
        dict(config)
    )

    # ========================================================
    # Data
    # ========================================================

    train_loader = build_decoder_dataloader(
        config,
        split="train",
    )

    val_loader = build_decoder_dataloader(
        config,
        split="val",
    )

    # ========================================================
    # Models
    # ========================================================

    encoder = build_encoder(
        config,
        device,
    )

    decoder = build_decoder(
        config,
        device,
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = build_optimizer(
        decoder,
        config["optimizer"],
    )

    # ========================================================
    # Scheduler
    #
    # Your existing build_scheduler() expects total_steps.
    # ========================================================

    steps_per_epoch =  config["training"]["steps_per_epoch"]

    total_steps = (
        steps_per_epoch
        * config["training"]["epochs"]
    )

    scheduler = build_scheduler(
        optimizer,
        config["scheduler"],
        total_steps,
    )

    # ========================================================
    # Checkpoint manager
    # ========================================================

    checkpoint_manager = CheckpointManager(
        output_dir / "checkpoints",
        keep_last=3,
    )

    # ========================================================
    # Training
    # ========================================================

    global_step = 0

    best_val_loss = float("inf")

    num_epochs = config["training"]["epochs"]

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        train_loss, global_step = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            device=device,
            epoch=epoch,
            global_step=global_step,
            config=config,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, visualization = validate(
            encoder=encoder,
            decoder=decoder,
            loader=val_loader,
            device=device,
            image_size=config["model"]["img_size"],
        )

        # ----------------------------------------------------
        # Epoch logging
        # ----------------------------------------------------

        logger.log(
            global_step,
            {
                "epoch/train_loss": train_loss,
                "epoch/val_loss": val_loss,
            },
        )

        print(
            f"\nEpoch {epoch}/{num_epochs}"
        )

        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Val loss:   {val_loss:.6f}"
        )

        # ----------------------------------------------------
        # W&B visualization
        # ----------------------------------------------------

        if visualization is not None:

            logger.log(
                global_step,
                {
                    "validation/input": wandb.Image(
                        visualization["input"],
                        caption="Original Bridge frame",
                    ),

                    "validation/reconstruction": wandb.Image(
                        visualization["reconstruction"],
                        caption="DINO decoder reconstruction",
                    ),
                },
            )

        # ----------------------------------------------------
        # Save latest
        # ----------------------------------------------------

        checkpoint_manager.save(
            model=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            step=global_step,
            loss=val_loss,
            config=dict(config),
            name="latest.pt",
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_manager.save(
                model=decoder,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=global_step,
                loss=val_loss,
                config=dict(config),
                name="best.pt",
            )

            print(
                "New best decoder checkpoint saved."
            )

    print(
        "\nDecoder training complete."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    main(
        args.config
    )