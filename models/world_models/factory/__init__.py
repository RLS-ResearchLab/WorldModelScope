from models.world_models.factory.dino_wm_builder import build_dino_wm

# As you implement each new model, import its builder here
# and add one line to the registry below. Nothing else in
# the codebase needs to change.
#
# from models.factory.vjepa_builder import build_vjepa
# from models.factory.hwm_builder import build_hwm

BUILDERS = {
    "dino_wm": build_dino_wm,
    # "vjepa": build_vjepa,
    # "hwm": build_hwm,
}


def build_model(config):
    """
    Single entry point for model construction.

    config["model"]["name"] selects which builder runs.
    Every builder receives the *entire* config (not just
    config["model"]) so it can also read config["training"]
    (e.g. device) if needed — same contract for every model.
    """

    name = config["model"]["name"]

    if name not in BUILDERS:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(BUILDERS.keys())}"
        )

    return BUILDERS[name](config)