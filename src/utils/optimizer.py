import torch


def build_optimizer(model, config):
    """
    Build optimizer using only trainable parameters.
    """

    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    name = config["name"].lower()

    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0)

    if name == "adamw":

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=config.get(
                "betas",
                (0.9, 0.999)
            ),
        )

    elif name == "adam":

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
            betas=config.get(
                "betas",
                (0.9, 0.999)
            ),
        )

    elif name == "sgd":

        optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=config.get(
                "momentum",
                0.9
            ),
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {name}"
        )

    return optimizer