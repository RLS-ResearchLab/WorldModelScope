import time
from pathlib import Path

import torch
from tqdm import tqdm

from src.utils.checkpoints import save_checkpoint, load_checkpoint


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        optimizers,
        schedulers=None,
        val_loader=None,
        logger=None,
        checkpoint_dir=None,
        config=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizers = optimizers or {}
        self.schedulers = schedulers or {}

        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or {}

        # ==========================================================
        # TRAINING CONFIG
        # ==========================================================

        training_cfg = self.config.get("training", {})

        self.precision = training_cfg.get("precision", "fp32")
        self.grad_accumulation_steps = training_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = training_cfg.get("max_grad_norm", None)
        self.val_every = training_cfg.get("val_every", 1000)  
        self.save_every = training_cfg.get("save_every", 1000) 
        self.log_every = training_cfg.get("log_every", 50)     

        self.max_steps = training_cfg.get("max_steps")
        if self.max_steps is None:
            raise ValueError(
                "config['training']['max_steps'] is required -- there's no epoch count to "
                "fall back on to know when training ends."
            )

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

        self.global_step = 0
        self.wandb_run_id = None  

        self.history = []

    # ==============================================================
    # BATCH PREP
    # ==============================================================

    def prepare_batch(self, raw_batch):
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
        actions_padded = torch.cat([actions, pad], dim=1)  # [B, T, A]

        return {
            "observations": frames,
            "actions": actions_padded,
        }

    def _next_batch(self, data_iter):
        """Pull the next batch, silently restarting the loader when the stream runs dry.

        BridgeDataset reshuffles internally each time it's re-iterated (shuffle=True for
        split="train"), so this is effectively "keep streaming, freshly shuffled" -- there's no
        epoch boundary to report, it's just how we keep the pipe full past one pass over
        whatever the loader currently has buffered.
        """
        try:
            return next(data_iter), data_iter
        except StopIteration:
            data_iter = iter(self.train_loader)
            return next(data_iter), data_iter

    

    def train(self):

        self.model.train()
        self._zero_grad()

        data_iter = iter(self.train_loader)

        progress = tqdm(total=self.max_steps, initial=self.global_step, desc="steps")
        total_start = time.time()

        while self.global_step < self.max_steps:

            # ------------------------------------------------------
            # ACCUMULATE grad_accumulation_steps micro-batches into
            # one optimizer step
            # ------------------------------------------------------

            last_loss = 0.0
            last_metrics = {}

            for _ in range(self.grad_accumulation_steps):

                raw_batch, data_iter = self._next_batch(data_iter)
                batch = self._move_batch(raw_batch)
                batch = self.prepare_batch(batch)

                with torch.autocast(
                    device_type=self.device_type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    output = self.model.compute_loss(batch)

                    if isinstance(output, tuple):
                        loss, metrics = output
                    else:
                        loss = output
                        metrics = {}

                    loss_for_backward = loss / self.grad_accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                # Accumulate (not overwrite) so the logged loss is the true average across all
                # micro-batches that contributed to this optimizer step, not just the last one.
                last_loss += loss.detach().item() / self.grad_accumulation_steps
                last_metrics = metrics

            # ------------------------------------------------------
            # ONE OPTIMIZER STEP == ONE TRAINING STEP
            # ------------------------------------------------------

            self._optimizer_step()
            self.global_step += 1
            progress.update(1)

            # ------------------------------------------------------
            # LOGGING (every log_every steps)
            # ------------------------------------------------------

            if self.global_step % self.log_every == 0:

                log_data = {
                    "train/loss": last_loss,
                    "train/lr": self._get_learning_rate(),
                    "time/total_seconds": time.time() - total_start,
                }

                for name, value in last_metrics.items():
                    if torch.is_tensor(value):
                        value = value.detach().float().mean().item()
                    elif isinstance(value, (int, float)):
                        value = float(value)
                    else:
                        continue
                    log_data[f"train/{name}"] = value

                if self.logger is not None:
                    self.logger.log(self.global_step, log_data)

                progress.set_postfix(loss=f"{last_loss:.4f}")
                self.history.append(log_data)

            # ------------------------------------------------------
            # VALIDATION (every val_every steps)
            # ------------------------------------------------------

            if self.val_loader is not None and self.global_step % self.val_every == 0:
                val_metrics = self.validate()
                if self.logger is not None:
                    self.logger.log(self.global_step, val_metrics)
                print(f"[step {self.global_step}] val/loss: {val_metrics.get('val/loss'):.6f}")

            # ------------------------------------------------------
            # CHECKPOINT (every save_every steps)
            # ------------------------------------------------------

            if self.checkpoint_dir is not None and self.global_step % self.save_every == 0:
                self._save_checkpoint(loss=last_loss)

        # final checkpoint at the very end, even if global_step doesn't land on save_every
        if self.checkpoint_dir is not None:
            self._save_checkpoint(loss=last_loss)

        total_time = time.time() - total_start
        print(f"Training completed in {total_time / 60:.2f} minutes ({self.global_step} steps)")

        return self.history

    # ==============================================================
    # OPTIMIZER STEP
    # ==============================================================

    def _optimizer_step(self):

        if self.max_grad_norm is not None:
            if self.scaler is not None:
                for optimizer in self.optimizers.values():
                    self.scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        for optimizer in self.optimizers.values():
            if self.scaler is not None:
                self.scaler.step(optimizer)
            else:
                optimizer.step()

        if self.scaler is not None:
            self.scaler.update()

        self._zero_grad()

        for scheduler in self.schedulers.values():
            scheduler.step()

    # ==============================================================
    # VALIDATION
    # ==============================================================

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        metrics_sum = {}

        # NOTE: val_loader may also be an IterableDataset -- iterating it fully here (rather
        # than a fixed number of steps) matches whatever "one validation pass" the dataset
        # itself defines (e.g. a held-out fixed-size split, unlike the infinite train stream).
        for batch in tqdm(self.val_loader, desc=f"validation @ step {self.global_step}"):

            batch = self._move_batch(batch)
            batch = self.prepare_batch(batch)

            with torch.autocast(
                device_type=self.device_type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = self.model.validation_step(batch)

                if isinstance(output, tuple):
                    loss, metrics = output
                else:
                    loss = output
                    metrics = {}

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

        return val_metrics

    # ==============================================================
    # CHECKPOINTING
    # ==============================================================

    def _save_checkpoint(self, loss):
        path = self.checkpoint_dir / f"step_{self.global_step:07d}.pt"
        save_checkpoint(
            path=path,
            step=self.global_step,
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            grad_scaler=self.scaler,
            loss=loss,
            wandb_run_id=self.wandb_run_id,
            config=self.config,
        )
        if self.logger is not None and hasattr(self.logger, "save_checkpoint"):
            self.logger.save_checkpoint(path, name=f"step-{self.global_step}-checkpoint")

    def load_checkpoint(self, path):
        """Resume training from a checkpoint. Restores model, optimizers, schedulers, AMP
        scaler, global step, and the wandb run id (so a resumed run can continue logging into
        the same wandb curve instead of starting a new disconnected one)."""

        resumed = load_checkpoint(
            path=path,
            model=self.model,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            grad_scaler=self.scaler,
            device=self.device,
        )

        self.global_step = resumed["step"]
        self.wandb_run_id = resumed["wandb_run_id"]

        print(f"Resumed from checkpoint: {path}")
        print(f"  global_step = {self.global_step}")
        print(f"  loss        = {resumed['loss']}")
        print(f"  wandb_run_id = {self.wandb_run_id}")

    # ==============================================================
    # HELPERS
    # ==============================================================

    def _get_learning_rate(self):
        if not self.optimizers:
            return 0.0
        optimizer = next(iter(self.optimizers.values()))
        return optimizer.param_groups[0]["lr"]

    def _zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def _move_batch(self, batch):
        if torch.is_tensor(batch):
            return batch.to(self.device, non_blocking=True)
        if isinstance(batch, dict):
            return {key: self._move_batch(value) for key, value in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch(value) for value in batch)
        return batch