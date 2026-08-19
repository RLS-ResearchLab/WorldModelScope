import torch 
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def build_scheduler(optimizer, config, max_steps):

    warmup_steps = config["warmup_steps"]

    warmup = LinearLR(
        optimizer,
        start_factor=1e-3,   # LR starts at ~0.1% of base LR
        end_factor=1.0,      # ramps up to 100% of base LR
        total_iters=warmup_steps,
    )

    decay = CosineAnnealingLR(
        optimizer,
        T_max=max_steps - warmup_steps,   # remaining steps after warmup
        eta_min=config.get("min_lr", 0.0),
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],   # switch from warmup → decay at this step
    )