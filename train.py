import random
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

    def _sample_window(self, frames, actions, window_len):
        """
        Slice a random contiguous window of `window_len` observations
        (and window_len - 1 aligned actions) out of a longer loaded clip.

        This is what decouples the dataset's clip_len (which can stay
        large, e.g. 24, for temporal diversity across a training run)
        from the predictor's fixed num_hist (which sets sequence length
        N = num_hist * (P+1), and therefore memory). Every step, the
        model only ever sees a window_len-length slice -- exactly the
        same amount of work regardless of how long the source clip is.

        frames:  [B, clip_len, C, H, W]
        actions: [B, clip_len - 1, A]

        Returns observations/actions of length window_len /
        window_len - 1, padded to window_len with a trailing zero
        action row (matching the original prepare_batch contract).
        """
        clip_len = frames.shape[1]

        if clip_len < window_len:
            raise ValueError(
                f"clip_len ({clip_len}) is shorter than the requested "
                f"window ({window_len} = num_hist + 1). Increase the "
                f"dataset's clip_len or decrease num_hist."
            )

        max_start = clip_len - window_len
        start = random.randint(0, max_start) if max_start > 0 else 0

        obs = frames[:, start:start + window_len]
        act = actions[:, start:start + window_len - 1]

        pad = torch.zeros_like(act[:, :1])
        act_padded = torch.cat([act, pad], dim=1)

        return {"observations": obs, "actions": act_padded}

    def prepare_batch(self, raw_batch):
        """
        raw_batch:
        frames:  [B, T, C, H, W]
        actions: [B, T-1, A]

        Draws a random num_hist+1 window from the (possibly longer)
        loaded clip. See `_sample_window` for why.
        """
        frames = raw_batch["frames"]
        actions = raw_batch["actions"]

        window_len = self.model.num_hist + 1
        return self._sample_window(frames, actions, window_len)

    def _iter_windows(self, raw_batch, window_len, stride=None):
        """
        Non-overlapping (by default) windows spanning the *entire*
        loaded clip, used for validation. Unlike `_sample_window`
        (one random slice, used for training), this walks the whole
        clip so a single long val clip yields several independent
        num_hist+1 evaluations -- more validation signal per clip
        without ever increasing the sequence length the model sees
        in one forward pass.
        """
        frames = raw_batch["frames"]
        actions = raw_batch["actions"]
        clip_len = frames.shape[1]

        if stride is None:
            stride = window_len - 1  # non-overlapping by default

        if clip_len < window_len:
            raise ValueError(
                f"Val clip_len ({clip_len}) is shorter than num_hist+1 "
                f"({window_len}); cannot form even one window."
            )

        for start in range(0, clip_len - window_len + 1, stride):
            obs = frames[:, start:start + window_len]
            act = actions[:, start:start + window_len - 1]
            pad = torch.zeros_like(act[:, :1])
            act_padded = torch.cat([act, pad], dim=1)
            yield {"observations": obs, "actions": act_padded}

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
                raw_batch = self._move_batch(raw_batch)
                batch = self.prepare_batch(raw_batch)

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
        """
        Validation clips are typically much longer than num_hist+1
        (e.g. clip_len=128 while num_hist=16), on purpose -- longer
        clips give more windows and thus a more reliable validation
        signal per loaded clip. Rather than requiring clip_len ==
        num_hist+1 (which would either crash or waste most of the
        loaded clip), we walk each val clip in non-overlapping
        num_hist+1 windows and average the loss across all of them.
        Each individual forward pass is still exactly num_hist+1
        frames, so this costs no more memory than a single training
        step -- it just does several such steps per loaded val clip.
        """
        self.model.eval()

        window_len = self.model.num_hist + 1

        total_loss = 0.0
        num_windows = 0
        metrics_sum = {}

        for raw_batch in tqdm(self.val_loader, desc=f"validation @ step {self.global_step}"):

            raw_batch = self._move_batch(raw_batch)

            for batch in self._iter_windows(raw_batch, window_len):

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
                num_windows += 1

                for name, value in metrics.items():
                    if torch.is_tensor(value):
                        value = value.detach().float().mean().item()
                    metrics_sum[name] = metrics_sum.get(name, 0.0) + value

        self.model.train()

        val_metrics = {"val/loss": total_loss / max(num_windows, 1)}
        for name, value in metrics_sum.items():
            val_metrics[f"val/{name}"] = value / max(num_windows, 1)
        val_metrics["val/num_windows"] = num_windows

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