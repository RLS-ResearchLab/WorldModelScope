import torch 
def build_scheduler(
    optimizer,
    config,
    total_steps,
):
    name = config.get(
        "name",
        "cosine"
    ).lower()

    if name == "cosine":

        min_lr = config.get(
            "min_lr",
            0.0
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=min_lr,
        )

    elif name == "constant":

        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=total_steps,
        )

    elif name == "step":

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config.get("gamma", 0.1),
        )

    else:
        raise ValueError(
            f"Unknown scheduler: {name}"
        )

    return scheduler