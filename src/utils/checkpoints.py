from pathlib import Path
import torch


class CheckpointManager:

    def __init__(
        self,
        directory,
        keep_last=3,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.keep_last = keep_last

    def save(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        step,
        loss,
        scaler=None,
        config=None,
        name="latest.pt",
    ):
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "loss": loss,

            "model": model.state_dict(),

            "optimizer": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),

            "scheduler": (
                scheduler.state_dict()
                if scheduler is not None
                else None
            ),

            "scaler": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),

            "config": config,
        }

        path = self.directory / name

        torch.save(
            checkpoint,
            path,
        )

        return path

    def load(
        self,
        path,
        model,
        optimizer=None,
        scheduler=None,
        scaler=None,
        device="cpu",
    ):

        checkpoint = torch.load(
            path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        if (
            optimizer is not None
            and checkpoint.get("optimizer") is not None
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        if (
            scheduler is not None
            and checkpoint.get("scheduler") is not None
        ):
            scheduler.load_state_dict(
                checkpoint["scheduler"]
            )

        if (
            scaler is not None
            and checkpoint.get("scaler") is not None
        ):
            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        return {
            "epoch": checkpoint.get(
                "epoch",
                0,
            ),
            "step": checkpoint.get(
                "step",
                0,
            ),
            "loss": checkpoint.get(
                "loss",
                None,
            ),
            "config": checkpoint.get(
                "config",
                None,
            ),
        }