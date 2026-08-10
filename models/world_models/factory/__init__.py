from models.world_models.factory.dino_wm_builder import build_dino_wm




BUILDERS = {
    "dino_wm": build_dino_wm,
    # "vjepa": build_vjepa,
    # "hwm": build_hwm,
}


def build_model(config):
    

    name = config["model"]["name"]

    if name not in BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(BUILDERS.keys())}"
        )

    return BUILDERS[name](config)