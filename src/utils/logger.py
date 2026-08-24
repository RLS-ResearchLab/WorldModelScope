from pathlib import Path
import csv
import json
import time

import wandb


class Logger:

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.output_dir / "metrics.csv"
        self._initialized = False

    def log(self, step, metrics):
        row = {"step": step, **metrics}

        print(
            " | ".join(
                f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in row.items()
            )
        )

        # Different callers (e.g. a training step vs. a validation pass) log different-shaped
        # metric dicts. Keep one fixed, growing column schema so rows always line up under the
        # header instead of being written positionally under whatever fieldnames the first-ever
        # row happened to have.
        if not self.log_file.exists():
            with open(self.log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)
            return

        with open(self.log_file, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            existing_rows = list(reader)

        new_keys = [key for key in row.keys() if key not in fieldnames]
        if new_keys:
            fieldnames = fieldnames + new_keys
            with open(self.log_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(row)
        else:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
                writer.writerow(row)

    def save_config(self, config):
        path = self.output_dir / "config.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=4)


class WandbLogger:
    def __init__(
        self,
        project,
        name,
        output_dir,
        config=None,
        group=None,
        tags=None,
        resume_id=None,
    ):
        """
        resume_id: pass the wandb_run_id read out of a checkpoint (via
        checkpoint.peek_wandb_run_id) to continue logging into that same run instead of
        starting a new, disconnected one. Leave None for a fresh run.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if resume_id is not None:
            # Resuming: reconnect to the exact same run. Don't touch `name` -- wandb keeps the
            # original run's name/history, a new timestamped name here would be misleading.
            self.run = wandb.init(
                project=project,
                id=resume_id,
                resume="must",
                dir=str(self.output_dir),
            )
        else:
            # Fresh run: only timestamp the name if one was actually given -- `f"{None}-..."`
            # produced a literal "None-<timestamp>" run name before this fix.
            run_name = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}" if name else None
            self.run = wandb.init(
                project=project,
                name=run_name,
                dir=str(self.output_dir),
                config=config,
                group=group,
                tags=tags,
                reinit="create_new",
            )

        # Exposed so callers (main.py) can pull the run id back out and hand it to Trainer for
        # checkpointing -- this is the other half of the round trip that resume_id closes.
        self.run_id = self.run.id
        print(f"W&B run initialized: {self.run.id}")

    def log(self, step, metrics):
        self.run.log(metrics, step=step)

    def save_checkpoint(self, checkpoint_path, name="model-checkpoint"):
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(str(checkpoint_path))
        self.run.log_artifact(artifact)


class CombinedLogger:
    def __init__(self, loggers):
        self.loggers = loggers

    def log(self, step, metrics):
        for logger in self.loggers:
            logger.log(step, metrics)

    def save_config(self, config):
        for logger in self.loggers:
            if hasattr(logger, "save_config"):
                logger.save_config(config)

    def save_checkpoint(self, checkpoint_path, name="model-checkpoint"):
        for logger in self.loggers:
            if hasattr(logger, "save_checkpoint"):
                logger.save_checkpoint(checkpoint_path, name=name)

    def get_wandb_run_id(self):
        """Returns the active wandb run id (from whichever sub-logger has one), so main.py can
        hand it to Trainer for checkpointing without needing to know WandbLogger sits inside a
        CombinedLogger."""
        for logger in self.loggers:
            if hasattr(logger, "run_id"):
                return logger.run_id
        return None