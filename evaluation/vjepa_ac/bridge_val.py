"""Deterministic held-out clips from the 80 BridgeData V2 validation shards.

Reads the official RLDS/TFDS release directly from `raw/bridge_dataset/1.0.0`.
Only the first 80 val shards are present locally, so the split is pinned to
`val[:4295]` -- exactly those 80 shards, and exactly the slice the training
script uses for its held-out eval (`heldout_next_latent_smooth_l1`).

Clip geometry matches the training script: 16 frames resampled from an assumed
5 Hz source to 4 fps (`idx = start + floor(arange(16) * 1.25)`), one `image_0`
view, native 7-D BridgeData action/state. Actions are taken at `idx[1::2]` and
states at `idx[::2]` -> 8 values each, one per encoder latent frame. One clip
per episode, random start, seeded for reproducibility.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

# first 80 val shards == first 4295 examples (see raw/.../dataset_info.json)
VAL_SLICE = "val[:4295]"

_SRC_FPS = 5.0
_TGT_FPS = 4.0
_CLIP_LEN = 16
_LAST_OFFSET = int(np.floor((_CLIP_LEN - 1) * _SRC_FPS / _TGT_FPS))   # 18
_MIN_STEPS = _LAST_OFFSET + 2                                          # 20


class BridgeValClips(IterableDataset):
    def __init__(
        self,
        data_root: str = "raw/bridge_dataset/1.0.0",
        split: str = VAL_SLICE,
        max_clips: int | None = None,
        camera_key: str = "image_0",
        seed: int = 0,
    ):
        self.data_root = str(data_root)
        self.split = split
        self.max_clips = max_clips
        self.camera_key = camera_key
        self.seed = seed

    def __iter__(self):
        import tensorflow as tf
        import tensorflow_datasets as tfds
        tf.config.set_visible_devices([], "GPU")

        info = torch.utils.data.get_worker_info()
        if info is not None and info.num_workers > 1:
            raise RuntimeError("BridgeValClips streams one tfds pipeline; use num_workers=0")

        builder = tfds.builder_from_directory(self.data_root)
        ds = builder.as_dataset(split=self.split, shuffle_files=False)
        rng = random.Random(self.seed)
        offsets = np.floor(np.arange(_CLIP_LEN) * _SRC_FPS / _TGT_FPS).astype(np.int64)

        n = 0
        for ep in tfds.as_numpy(ds):
            steps = list(ep["steps"])
            if len(steps) < _MIN_STEPS:
                continue
            im = np.stack([s["observation"][self.camera_key] for s in steps])
            ac = np.stack([s["action"] for s in steps]).astype(np.float32)
            st = np.stack([s["observation"]["state"] for s in steps]).astype(np.float32)

            start = rng.randrange(0, len(im) - _LAST_OFFSET - 1)
            idx = start + offsets
            yield {
                "frames": torch.from_numpy(im[idx].copy()),          # [16, H, W, 3] uint8
                "actions": torch.from_numpy(ac[idx[1::2]].copy()),   # [8, 7]
                "states": torch.from_numpy(st[idx[::2]].copy()),     # [8, 7]
            }
            n += 1
            if self.max_clips and n >= self.max_clips:
                return
