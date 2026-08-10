from pathlib import Path
from omegaconf import OmegaConf


def load_config(config_path):
    """
    Load a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration.

    Returns:
        OmegaConf configuration object.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    if config_path.suffix not in [".yaml", ".yml"]:
        raise ValueError(
            f"Expected a YAML file, got: {config_path.suffix}"
        )

    config = OmegaConf.load(config_path)

    return config