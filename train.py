
import torch
from tqdm import tqdm
import time


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        optimizers,
        schedulers=None,
        val_loader=None,
        logger=None,
        checkpoint_manager=None,
        config=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizers = optimizers or {}
        self.schedulers = schedulers or {}

        self.logger = logger
        self.checkpoint_manager = checkpoint_manager

        self.config = config or {}
        

        # ==========================================================
        # TRAINING CONFIG
        # ==========================================================

        training_cfg = self.config.get("training", {})

        self.num_epochs = training_cfg.get("epochs", 1)
        self.precision = training_cfg.get("precision", "fp32")
        self.grad_accumulation_steps = training_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = training_cfg.get("max_grad_norm", None)
        self.val_every = training_cfg.get("val_every", 1)
        self.save_every = training_cfg.get("save_every", 1)
        self.log_every = training_cfg.get("log_every", 1)
        self.warmup_steps = training_cfg.get("warmup_steps")
        self.steps_per_epoch = training_cfg.get("steps_per_epoch",None,)


        # DEVICE
        self.device = torch.device(
        training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
        self.device_type = self.device.type

        self.use_amp = self.device.type == "cuda" and self.precision != "fp32"

        if self.precision == "bf16":
            self.amp_dtype = torch.bfloat16
        elif self.precision == "fp16":
            self.amp_dtype = torch.float16
        else:
            self.amp_dtype = torch.float32

        if self.precision == "fp16" and self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # ==========================================================
        # RESUME STATE
        # ==========================================================

        self.start_epoch = 0
        self.global_step = 0

        self.history = []

    # ==============================================================
    # TRAIN ONE EPOCH
    # ==============================================================
   
    def prepare_batch(self,raw_batch):
        """
        raw_batch:
        frames:  [B, T, C, H, W]
        actions: [B, T-1, A]

        returns:
        observations: [B, T, C, H, W]
        actions:      [B, T, A]        (last row is a zero pad, unused in loss)
        """
        frames = raw_batch["frames"]
        actions = raw_batch["actions"]

        B, T_minus_1, A = actions.shape
        pad = torch.zeros(B, 1, A, dtype=actions.dtype, device=actions.device)
        actions_padded = torch.cat([actions, pad], dim=1)   # [B, T, A]

        return {
            "observations": frames,
            "actions": actions_padded,
        }
    def train_epoch(self, epoch):

        self.model.train()

        total_loss = 0.0
        num_batches = 0

        self._zero_grad()

        progress = tqdm(
            self.train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{self.num_epochs}"
            ),
        )

        for batch_idx, batch in enumerate(progress):

            batch = self._move_batch(batch)
            batch = self.prepare_batch(batch)

            with torch.autocast(
                device_type=self.device_type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):

                output = self.model.compute_loss(
                    batch
                )

            

                if isinstance(output, tuple):

                    loss, metrics = output

                else:

                    loss = output
                    metrics = {}

                loss_for_backward = (
                    loss
                    / self.grad_accumulation_steps
                )

            # ======================================================
            # BACKWARD
            # ======================================================

            if self.scaler is not None:

                self.scaler.scale(
                    loss_for_backward
                ).backward()

            else:

                loss_for_backward.backward()

            total_loss += loss.detach().item()
            num_batches += 1

            # ======================================================
            # ACCUMULATION
            # ======================================================

            should_step = (
                (batch_idx + 1)
                % self.grad_accumulation_steps
                == 0
            )

            is_last_batch = (
            self.steps_per_epoch is not None
            and batch_idx + 1 >= self.steps_per_epoch
        )

            if should_step or is_last_batch:

                self._optimizer_step()

                self.global_step += 1

                # ==================================================
                # LOGGING
                # ==================================================

                log_data = {
                    "train/loss": (
                        loss.detach().item()
                    ),
                    "train/lr": (
                        self._get_learning_rate()
                    ),
                }

                for name, value in metrics.items():

                    if torch.is_tensor(value):

                        value = (
                            value.detach()
                            .float()
                            .mean()
                            .item()
                        )

                    elif isinstance(
                        value,
                        (int, float),
                    ):

                        value = float(value)

                    else:

                        continue

                    log_data[
                        f"train/{name}"
                    ] = value

                if self.logger is not None:

                    self.logger.log(
                        self.global_step,
                        log_data,
                    )

                progress.set_postfix(
                    loss=(
                        f"{loss.detach().item():.4f}"
                    )
                )
            if (
                self.steps_per_epoch is not None
                and batch_idx + 1 >= self.steps_per_epoch
            ):
                break

        epoch_loss = (
            total_loss
            / max(num_batches, 1)
        )

        return epoch_loss

    # ==============================================================
    # OPTIMIZER STEP
    # ==============================================================

    def _optimizer_step(self):

        # ----------------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------------

        if self.max_grad_norm is not None:

            # FP16 gradients are scaled.
            # Unscale before clipping.
            if self.scaler is not None:

                for optimizer in (
                    self.optimizers.values()
                ):

                    self.scaler.unscale_(
                        optimizer
                    )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

        # ----------------------------------------------------------
        # Optimizers
        # ----------------------------------------------------------

        for optimizer in (
            self.optimizers.values()
        ):

            if self.scaler is not None:

                self.scaler.step(
                    optimizer
                )

            else:

                optimizer.step()

       
        if self.scaler is not None:

            self.scaler.update()

       
        self._zero_grad()

        
        for scheduler in (
            self.schedulers.values()
        ):

            scheduler.step()

    # ==============================================================
    # VALIDATION
    # ==============================================================

    @torch.no_grad()
    def validate(self, epoch):

        if self.val_loader is None:

            return {}

        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        metrics_sum = {}

        progress = tqdm(
            self.val_loader,
            desc=(
                f"Validation "
                f"{epoch + 1}/{self.num_epochs}"
            ),
        )

        for batch in progress:

            batch = self._move_batch(batch)

            with torch.autocast(
                device_type=self.device_type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):

                output = (
                    self.model.validation_step(
                        batch
                    )
                )

                if isinstance(output, tuple):

                    loss, metrics = output

                else:

                    loss = output
                    metrics = {}

            total_loss += (
                loss.detach().item()
            )

            num_batches += 1

            for name, value in metrics.items():

                if torch.is_tensor(value):

                    value = (
                        value.detach()
                        .float()
                        .mean()
                        .item()
                    )

                if name not in metrics_sum:

                    metrics_sum[name] = 0.0

                metrics_sum[name] += value

        val_loss = (
            total_loss
            / max(num_batches, 1)
        )

        metrics = {
            "val/loss": val_loss
        }

        for name, value in metrics_sum.items():

            metrics[
                f"val/{name}"
            ] = (
                value
                / max(num_batches, 1)
            )

        return metrics

    # ==============================================================
    # FIT
    # ==============================================================

    def fit(self):
        total_start = time.time()

        for epoch in range(
            self.start_epoch,
            self.num_epochs,
        ):

            
            epoch_start = time.time()

            train_loss = (
                self.train_epoch(epoch)
            )
            epoch_time = time.time() - epoch_start
            total_time = time.time() - total_start
            

            epoch_metrics = {
                "epoch": epoch + 1,
                "train/epoch_loss": train_loss,
                "time/epoch_seconds": epoch_time,
                "time/total_seconds": total_time,
            }

            # ======================================================
            # VALIDATION
            # ======================================================

            if (
                self.val_loader is not None
                and (
                    (epoch + 1)
                    % self.val_every
                    == 0
                )
            ):

                val_metrics = (
                    self.validate(epoch)
                )

                epoch_metrics.update(
                    val_metrics
                )

            # ======================================================
            # LOG EPOCH
            # ======================================================

            if self.logger is not None:

                self.logger.log(
                    self.global_step,
                    epoch_metrics,
                )

            print(
                f"Epoch {epoch + 1}/"
                f"{self.num_epochs} "
                f"- train loss: "
                f"{train_loss:.6f}"
                f"- time: {epoch_time:.2f}s"
            )

            if "val/loss" in epoch_metrics:

                print(
                    f"  val loss: "
                    f"{epoch_metrics['val/loss']:.6f}"
                )

           

            if (
            self.checkpoint_manager is not None
        ):
                main_optimizer = next(iter(self.optimizers.values()), None)
                main_scheduler = next(iter(self.schedulers.values()), None)

                latest_path  = self.checkpoint_manager.save(
                    model=self.model,
                    optimizers=self.optimizers,
                    schedulers=self.schedulers,
                    epoch=epoch,
                    global_step=self.global_step,
                    loss=train_loss,
                    scaler=self.scaler,
                    config=self.config,
                    name="latest.pt",
                )
                # upload "latest" every epoch (cheap, always overwritten)
                if self.logger is not None and hasattr(self.logger, "save_checkpoint"):
                    self.logger.save_checkpoint(latest_path, name="latest-checkpoint")
                if (epoch + 1) % self.save_every == 0:
                    epoch_path  = self.checkpoint_manager.save(
                        model=self.model,
                        optimizers=self.optimizers,
                        schedulers=self.schedulers,
                        epoch=epoch,
                        global_step=self.global_step,
                        loss=train_loss,
                        scaler=self.scaler,
                        config=self.config,
                        name=f"epoch_{epoch + 1:04d}.pt",
                    )
                # upload periodic checkpoints (versioned artifacts)
                    if self.logger is not None and hasattr(self.logger, "save_checkpoint"):
                        self.logger.save_checkpoint(epoch_path, name="epoch-checkpoint")

            self.history.append(
                epoch_metrics
            )
        total_time = time.time() - total_start
        print(
    f"Training completed in "
    f"{total_time / 60:.2f} minutes"
)

        return self.history

    # ==============================================================
    # LEARNING RATE
    # ==============================================================

    def _get_learning_rate(self):

        if not self.optimizers:

            return 0.0

        optimizer = next(
            iter(
                self.optimizers.values()
            )
        )

        return optimizer.param_groups[0][
            "lr"
        ]

    # ==============================================================
    # ZERO GRADIENTS
    # ==============================================================

    def _zero_grad(self):

        for optimizer in (
            self.optimizers.values()
        ):

            optimizer.zero_grad(
                set_to_none=True
            )

    def _move_batch(self, batch):

        if torch.is_tensor(batch):

            return batch.to(
                self.device,
                non_blocking=True,
            )

        if isinstance(batch, dict):

            return {
                key: self._move_batch(value)
                for key, value in batch.items()
            }

        if isinstance(batch, (list, tuple)):

            return type(batch)(
                self._move_batch(value)
                for value in batch
            )

        return batch

    # ==============================================================
    # LOAD CHECKPOINT
    # ==============================================================

    def load_checkpoint(self, path):
        """
        Resume training from a checkpoint.

        Restores:
            - model
            - optimizers
            - schedulers
            - AMP scaler
            - epoch
            - global step
            - loss
            - training history
        """

        if self.checkpoint_manager is None:
            raise RuntimeError(
                "Cannot load checkpoint: "
                "checkpoint_manager is None."
            )

        checkpoint = self.checkpoint_manager.load(
            path=path,
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            scaler=self.scaler,
            device=self.device,
        )

        # ----------------------------------------------------------
        # RESUME STATE
        # ----------------------------------------------------------

        self.start_epoch = checkpoint.get(
            "epoch",
            0,
        )

        self.global_step = checkpoint.get(
            "global_step",
            0,
        )

        self.last_loss = checkpoint.get(
            "loss",
            None,
        )

        self.loaded_config = checkpoint.get(
            "config",
            None,
        )

        print(
            f"Resumed from checkpoint: {path}"
        )

        print(
            f"  epoch      = {self.start_epoch}"
        )

        print(
            f"  global_step = {self.global_step}"
        )

        print(
            f"  loss       = {self.last_loss}"
        )