"""Save/resume for dino_wm training: step-based (not epoch-based) checkpointing.

Why step-based: BridgeDataset is an IterableDataset streaming trajectory windows -- there's no
fixed-length "pass over the dataset" to call an epoch, so `step` (= number of optimizer updates)
is the only unit that stays well-defined regardless of how much data exists or how it's shuffled.
It's also what the LR scheduler and wandb x-axis already track.

Matches the real Trainer structure: one unified `model` (not separate submodules), plus dict-style
`optimizers` / `schedulers` (Trainer supports multiple named optimizers, e.g. different LR groups,
even though main.py currently only registers one under "main").
"""
from pathlib import Path
from typing import Any

import torch

def peek_wandb_run_id(path: str | Path) -> str | None:
    """Read only the W&B run ID stored in a checkpoint.

    Used before creating the W&B logger so a resumed training run
    reconnects to the exact same W&B run.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return ckpt.get("wandb_run_id")

def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("step_*.pt"))

    if not checkpoints:
        return None

    def get_step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    checkpoints = [
        path for path in checkpoints
        if get_step(path) >= 0
    ]

    if not checkpoints:
        return None

    return max(checkpoints, key=get_step)


def save_checkpoint(
    path: str | Path,
    step: int,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer],
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    loss: float | None = None,
    wandb_run_id: str | None = None,
    config: dict | None = None,
) -> None:
    """Atomic write (temp file + rename): a crash mid-save never leaves `path` pointing at a
    half-written file.

    Every checkpoint is kept -- no pruning here. Call this less often (bump `save_every` in
    config) rather than pruning after the fact, if disk becomes a concern.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
            "schedulers": (
                {name: sch.state_dict() for name, sch in schedulers.items()}
                if schedulers
                else {}
            ),
            "grad_scaler": grad_scaler.state_dict() if grad_scaler is not None else None,
            "loss": loss,
            "wandb_run_id": wandb_run_id,
            "config": config,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Loads every component in place. Returns a dict with step / loss / wandb_run_id / config
    for the caller to pull resume state from.

    Uses .get(...) throughout so a checkpoint saved before some field existed (e.g. no
    wandb_run_id yet) still loads fine -- missing just means None, same as if it were never
    passed at save time.
    """
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model"])

    if optimizers is not None:
        saved_optimizers = ckpt.get("optimizers", {})
        for name, optimizer in optimizers.items():
            if name in saved_optimizers:
                optimizer.load_state_dict(saved_optimizers[name])

    if schedulers is not None:
        saved_schedulers = ckpt.get("schedulers", {})
        for name, scheduler in schedulers.items():
            if name in saved_schedulers:
                scheduler.load_state_dict(saved_schedulers[name])

    if grad_scaler is not None and ckpt.get("grad_scaler") is not None:
        grad_scaler.load_state_dict(ckpt["grad_scaler"])

    return {
        "step": ckpt.get("step", 0),
        "loss": ckpt.get("loss"),
        "wandb_run_id": ckpt.get("wandb_run_id"),
        "config": ckpt.get("config"),
    }