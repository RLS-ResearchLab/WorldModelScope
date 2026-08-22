"""On-demand BridgeData V2 clip dataset and PyTorch DataLoaders.

The dataset reads the bounded BridgeData V2 TFDS/RLDS subset directly
from TFRecord shards and creates clips without materializing .npz files.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf

try:
    import torch
    from torch.utils.data import (
        DataLoader,
        IterableDataset,
        get_worker_info,
    )
except ImportError as exc:
    raise ImportError(
        "bridge_data.dataset requires PyTorch"
    ) from exc


# =========================================================
# IMPORTANT:
# Keep these synchronized with download.py
# =========================================================

TRAIN_SHARDS = 400
VAL_SHARDS = 80


def resample_indices(
    length: int,
    source_fps: float = 5.0,
    target_fps: float = 4.0,
) -> np.ndarray:

    if (
        length <= 0
        or source_fps <= 0
        or target_fps <= 0
    ):
        raise ValueError(
            "length and FPS values must be positive"
        )

    count = (
        int(
            np.floor(
                (length - 1)
                * target_fps
                / source_fps
            )
        )
        + 1
    )

    return np.unique(
        np.rint(
            np.arange(count)
            * source_fps
            / target_fps
        ).astype(np.int64)
    )


def _expected_shard_files(
    data_dir: Path,
    split: str,
) -> list[str]:

    if split == "train":
        count = TRAIN_SHARDS
        total = 1024

    elif split == "val":
        count = VAL_SHARDS
        total = 128

    else:
        raise ValueError(
            f"Unknown split: {split}"
        )

    expected = [
        data_dir
        / (
            f"bridge_dataset-{split}."
            f"tfrecord-{i:05d}-of-{total:05d}"
        )
        for i in range(count)
    ]

    # -----------------------------------------------------
    # CRITICAL:
    # Do not silently use whatever happens to be present.
    # Require exactly the bounded subset.
    # -----------------------------------------------------

    missing = [
        path
        for path in expected
        if not path.exists()
    ]

    if missing:

        preview = "\n".join(
            f"  - {path.name}"
            for path in missing[:20]
        )

        extra = (
            f"\n  ... and {len(missing) - 20} more"
            if len(missing) > 20
            else ""
        )

        raise FileNotFoundError(
            f"Missing {len(missing)} {split} shards.\n"
            f"Expected the first {count} {split} shards.\n"
            f"Missing:\n{preview}{extra}\n\n"
            "Run bridge_data/download.py first."
        )

    return [
        str(path)
        for path in expected
    ]


class BridgeDataset(IterableDataset):
    """Stream fixed-length clips from BridgeData V2.

    Each item contains:

        frames  [T,C,H,W]
        actions [T-1,7]
        states  [T,7]
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
        image_size: int = 224,          # NEW
        shuffle: bool | None = None,
    ) -> None:

        if split == "heldout":
            split = "val"

        if split not in {"train", "val"}:
            raise ValueError(
                "split must be 'train', 'val', or 'heldout'"
            )

        if clip_len < 2 or stride < 1:
            raise ValueError(
                "clip_len must be at least 2 "
                "and stride must be positive"
            )

        self.data_dir = Path(data_dir)
        self.split = split

        self.clip_len = clip_len
        self.stride = stride

        self.camera_key = camera_key

        self.source_fps = source_fps
        self.target_fps = target_fps

        self.image_size = image_size    # NEW

        self.shuffle = (
            split == "train"
            if shuffle is None
            else shuffle
        )

    def __iter__(self):

        try:
            import tensorflow_datasets as tfds

        except ImportError as exc:
            raise ImportError(
                "BridgeDataset requires tensorflow-datasets"
            ) from exc

        worker = get_worker_info()

        worker_id = (
            worker.id
            if worker
            else 0
        )

        worker_count = (
            worker.num_workers
            if worker
            else 1
        )

        rng = random.Random(
            torch.initial_seed()
            + worker_id
        )

        # -------------------------------------------------
        # IMPORTANT:
        # Explicitly use only the first 400/80 shards.
        # -------------------------------------------------

        shard_files = _expected_shard_files(
            self.data_dir,
            self.split,
        )

        if self.shuffle:
            rng.shuffle(shard_files)

        print(
            f"[BridgeDataset] "
            f"worker={worker_id}/{worker_count} "
            f"split={self.split} "
            f"shards={len(shard_files)}",
            flush=True,
        )

        # -------------------------------------------------
        # TFRecord reader
        # -------------------------------------------------

        raw_ds = tf.data.TFRecordDataset(
            shard_files,
            num_parallel_reads=1,
        )

        builder = tfds.builder_from_directory(
            str(self.data_dir)
        )

        decoded_ds = raw_ds.map(
            builder.info.features.deserialize_example,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # -------------------------------------------------
        # Episodes
        # -------------------------------------------------

        for episode_index, episode in enumerate(
            tfds.as_numpy(decoded_ds)
        ):

            if (
                episode_index % worker_count
                != worker_id
            ):
                continue

            steps = list(
                episode["steps"]
            )

            indices = resample_indices(
                len(steps),
                self.source_fps,
                self.target_fps,
            )

            starts = list(
                range(
                    0,
                    len(indices)
                    - self.clip_len
                    + 1,
                    self.stride,
                )
            )

            if self.shuffle:
                rng.shuffle(starts)

            for start in starts:

                selected = indices[
                    start:
                    start + self.clip_len
                ]

                frames = np.stack(
                    [
                        tf.image.resize(
                            steps[i]["observation"][self.camera_key],
                            [self.image_size, self.image_size],
                        ).numpy()
                        for i in selected
                    ]
)

                states = np.stack(
                    [
                        steps[i]
                        ["observation"]
                        ["state"]
                        for i in selected
                    ]
                ).astype(np.float32)

                actions = np.stack(
                    [
                        steps[i]["action"]
                        for i in selected[:-1]
                    ]
                ).astype(np.float32)

                frames = np.ascontiguousarray(
                    np.transpose(
                        frames,
                        (0, 3, 1, 2),
                    ),
                    dtype=np.float32,
                ) / 255.0

                yield {
                    "frames": torch.from_numpy(
                        frames
                    ),
                    "actions": torch.from_numpy(
                        actions
                    ),
                    "states": torch.from_numpy(
                        states
                    ),
                }


def make_dataloader(
    dataset: BridgeDataset,
    batch_size: int = 16,
    **kwargs,
) -> DataLoader:

    return DataLoader(
        dataset,
        batch_size=batch_size,
        **kwargs,
    )


def make_dataloaders(
    data_dir: str | Path = "raw/bridge_dataset/1.0.0",
    batch_size: int = 16,
    **kwargs,
) -> tuple[DataLoader, DataLoader]:

    train_kwargs = dict(kwargs)
    val_kwargs = dict(kwargs)

    # IterableDataset performs its own shuffling.
    train_kwargs.setdefault(
        "shuffle",
        False,
    )

    val_kwargs.setdefault(
        "shuffle",
        False,
    )

    train_loader = make_dataloader(
        BridgeDataset(
            data_dir,
            "train",
        ),
        batch_size,
        **train_kwargs,
    )

    val_loader = make_dataloader(
        BridgeDataset(
            data_dir,
            "val",
            shuffle=False,
        ),
        batch_size,
        **val_kwargs,
    )

    return train_loader, val_loader