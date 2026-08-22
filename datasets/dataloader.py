from pathlib import Path
from torch.utils.data import DataLoader

from datasets._temp.dataset import BridgeDataset

DATASET_REGISTRY = {
    "bridge": BridgeDataset,
}


def build_dataset(data_cfg, split, image_size=224):
    name = data_cfg["name"]
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASET_REGISTRY)}")
    params = dict(data_cfg.get("params", {}))
    split_cfg = data_cfg.get(split, {})
    params.update(split_cfg)
    params["image_size"] = image_size

    return DATASET_REGISTRY[name](split=split, **params)


def build_dataloader(config, split="train"):
    """
    Build a single split's DataLoader from config['data'].
    NOTE: BridgeDataset is an IterableDataset -> never pass shuffle= to DataLoader.
    Shuffling for train is handled internally by the dataset (shuffle=True by default when split='train').
    """
    data_cfg = config["data"]
    image_size = config["model"]["image_size"]
    dataset = build_dataset(data_cfg, split=split, image_size=image_size)

    return DataLoader(
        dataset,
        batch_size=data_cfg.get("batch_size", 16),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        # no `shuffle` kwarg here — IterableDataset doesn't support it
    )


def build_dataloaders(config):
    data_cfg = config["data"]

    """Convenience: return both train and val loaders."""
    return {
        "train": build_dataloader(config, split="train"),
        "val": build_dataloader(config, split="val"),
    }

    