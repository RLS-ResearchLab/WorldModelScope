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
        optimizers=None,
        schedulers=None,
        epoch=0,
        global_step=0,
        loss=None,
        scaler=None,
        config=None,
        name="latest.pt",
    ):
        """
        Save a complete training checkpoint.

        Stores:
            - model weights
            - optimizer states
            - scheduler states
            - AMP scaler state
            - epoch
            - global step
            - loss
            - config
        """

        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss,

            "model": model.state_dict(),

            "optimizers": {
                name: optimizer.state_dict()
                for name, optimizer in (optimizers or {}).items()
            },

            "schedulers": {
                name: scheduler.state_dict()
                for name, scheduler in (schedulers or {}).items()
            },

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

        # Keep only the requested number of epoch checkpoints.
        self._cleanup_old_checkpoints()

        return path

    def load(
    self,
    path,
    model,
    optimizers=None,
    schedulers=None,
    scaler=None,
    device="cpu",
):
        """
        Load a checkpoint.

        Supports both:

        OLD format:
            optimizer
            scheduler
            step

        NEW format:
            optimizers
            schedulers
            global_step
        """

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

        # ==========================================================
        # MODEL
        # ==========================================================

        model.load_state_dict(
            checkpoint["model"]
        )

        # ==========================================================
        # OPTIMIZERS
        # ==========================================================

        if optimizers is not None:

            # ------------------------------------------------------
            # NEW FORMAT
            # ------------------------------------------------------

            if "optimizers" in checkpoint:

                saved_optimizers = checkpoint["optimizers"]

                for name, optimizer in optimizers.items():

                    if name in saved_optimizers:

                        optimizer.load_state_dict(
                            saved_optimizers[name]
                        )

            # ------------------------------------------------------
            # OLD FORMAT
            # ------------------------------------------------------

            elif checkpoint.get("optimizer") is not None:

                optimizer_list = list(
                    optimizers.values()
                )

                if optimizer_list:

                    optimizer_list[0].load_state_dict(
                        checkpoint["optimizer"]
                    )

                    print(
                        "Loaded optimizer from old "
                        "checkpoint format."
                    )

        # ==========================================================
        # SCHEDULERS
        # ==========================================================

        if schedulers is not None:

            # ------------------------------------------------------
            # NEW FORMAT
            # ------------------------------------------------------

            if "schedulers" in checkpoint:

                saved_schedulers = checkpoint["schedulers"]

                for name, scheduler in schedulers.items():

                    if name in saved_schedulers:

                        scheduler.load_state_dict(
                            saved_schedulers[name]
                        )

            # ------------------------------------------------------
            # OLD FORMAT
            # ------------------------------------------------------

            elif checkpoint.get("scheduler") is not None:

                scheduler_list = list(
                    schedulers.values()
                )

                if scheduler_list:

                    scheduler_list[0].load_state_dict(
                        checkpoint["scheduler"]
                    )

                    print(
                        "Loaded scheduler from old "
                        "checkpoint format."
                    )

        # ==========================================================
        # AMP SCALER
        # ==========================================================

        if (
            scaler is not None
            and checkpoint.get("scaler") is not None
        ):

            scaler.load_state_dict(
                checkpoint["scaler"]
            )

        # ==========================================================
        # EPOCH
        # ==========================================================

        epoch = checkpoint.get(
            "epoch",
            0,
        )

        # ==========================================================
        # GLOBAL STEP
        # ==========================================================

        # New checkpoint format
        if "global_step" in checkpoint:

            global_step = checkpoint["global_step"]

        # Old checkpoint format
        else:

            global_step = checkpoint.get(
                "step",
                0,
            )

        # ==========================================================
        # RETURN
        # ==========================================================

        return {
            "epoch": epoch,

            "global_step": global_step,

            "loss": checkpoint.get(
                "loss",
                None,
            ),

            "config": checkpoint.get(
                "config",
                None,
            ),

            "history": checkpoint.get(
                "history",
                [],
            ),
        }

    def _cleanup_old_checkpoints(self):
        """
        Keep only the latest `keep_last` epoch checkpoints.

        `latest.pt` is always preserved.
        """

        if self.keep_last is None:
            return

        checkpoints = list(
            self.directory.glob("epoch_*.pt")
        )

        checkpoints.sort(
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for path in checkpoints[self.keep_last:]:
            path.unlink()