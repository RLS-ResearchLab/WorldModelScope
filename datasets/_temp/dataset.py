"""On-demand BridgeData V2 clip dataset and PyTorch DataLoaders.

The dataset reads downloaded TFDS/RLDS episodes directly and creates clips
without materializing an intermediate directory of ``.npz`` files.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

try: 
    import torch
    from torch.utils.data import DataLoader, IterableDataset, get_worker_info
except ImportError as exc:  # keep import errors useful for non-training code
    raise ImportError("bridge_data.dataset requires PyTorch") from exc


def resample_indices(length: int, source_fps: float = 5.0, target_fps: float = 4.0) -> np.ndarray:
    if length <= 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("length and FPS values must be positive")
    count = int(np.floor((length - 1) * target_fps / source_fps)) + 1
    return np.unique(np.rint(np.arange(count) * source_fps / target_fps).astype(np.int64))


class BridgeDataset(IterableDataset):
    """Stream fixed-length clips from one official BridgeData split.

    Each item is a dict with ``frames`` [T,C,H,W], ``actions`` [T-1,7], and
    ``states`` [T,7], all PyTorch tensors. ``split`` is ``"train"`` or
    ``"val"`` (``"heldout"`` is accepted as an alias for ``"val"``).
    """

    def __init__(
        self,
        data_dir: str | Path = "raw/bridge_dataset/1.0.0",
        split: str = "train",
        clip_len: int = 16,
        stride: int = 8,
        camera_key: str = "image_0",
        source_fps: float = 5.0,
        target_fps: float = 4.0,
        shuffle: bool | None = None,
    ) -> None:
        if split == "heldout":
            split = "val"
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train', 'val', or 'heldout'")
        if clip_len < 2 or stride < 1:
            raise ValueError("clip_len must be at least 2 and stride must be positive")
        self.data_dir, self.split = Path(data_dir), split
        self.clip_len, self.stride = clip_len, stride
        self.camera_key = camera_key
        self.source_fps, self.target_fps = source_fps, target_fps
        self.shuffle = split == "train" if shuffle is None else shuffle

    def __iter__(self):
        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise ImportError("BridgeDataset requires tensorflow-datasets") from exc

        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1
        rng = random.Random(torch.initial_seed() + worker_id)
        builder = tfds.builder_from_directory(str(self.data_dir))
        dataset = builder.as_dataset(split=self.split, shuffle_files=self.shuffle)

        for episode_index, episode in enumerate(tfds.as_numpy(dataset)):
            if episode_index % worker_count != worker_id:
                continue
            steps = list(episode["steps"])
            indices = resample_indices(len(steps), self.source_fps, self.target_fps)
            starts = range(0, len(indices) - self.clip_len + 1, self.stride)
            starts = list(starts)
            if self.shuffle:
                rng.shuffle(starts)
            for start in starts:
                selected = indices[start : start + self.clip_len]
                frames = np.stack([steps[i]["observation"][self.camera_key] for i in selected])
                states = np.stack([steps[i]["observation"]["state"] for i in selected]).astype(np.float32)
                actions = np.stack([steps[i]["action"] for i in selected[:-1]]).astype(np.float32)
                frames = np.ascontiguousarray(np.transpose(frames, (0, 3, 1, 2)), dtype=np.float32) / 255.0
                yield {
                    "frames": torch.from_numpy(frames),
                    "actions": torch.from_numpy(actions),
                    "states": torch.from_numpy(states),
                }


def make_dataloader(dataset: BridgeDataset, batch_size: int = 16, **kwargs) -> DataLoader:
    """Create a DataLoader from a ``BridgeDataset`` instance."""
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def make_dataloaders(
    data_dir: str | Path = "raw/bridge_dataset/1.0.0",
    batch_size: int = 16,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)`` over the downloaded subset."""
    train_kwargs = dict(kwargs)
    val_kwargs = dict(kwargs)
    train_kwargs.setdefault("shuffle", False)  # IterableDataset shuffles internally.
    val_kwargs.setdefault("shuffle", False)
    return (
        make_dataloader(BridgeDataset(data_dir, "train"), batch_size, **train_kwargs),
        make_dataloader(BridgeDataset(data_dir, "val", shuffle=False), batch_size, **val_kwargs),
    )

