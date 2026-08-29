"""The canonical clip stream -- Level A fairness (identical inputs).

One resampling scheme, one camera, one resolution, one seed. Every adapter
receives *the same bytes* and subsamples to its own geometry inside ``encode`` /
``align_actions``. No model gets to pick more favourable frames.

Canonical geometry
------------------
* source  : the official RLDS/TFDS BridgeData V2 release at
            ``raw/bridge_dataset/1.0.0`` (``image_0`` view, 7-D action + state).
* fps     : 5 Hz source -> 4 fps  (``idx = start + floor(arange(N) * 1.25)``),
            matching the frozen V-JEPA 2.1-AC checkpoint's training script.
* N       : ``CANON_FRAMES = 16`` -- the superset over all four models
            (V-JEPA 16 raw / 8 latent, DINO-WM 13, LeWM 12).
* clips   : one per episode, random start, seeded -> deterministic given ``seed``.

Two eval sets, always reported separately (Level E):
* ``val[:4295]``  -- the 80 val shards present locally; the training-time
                     held-out slice.
* ``val[4295:]``  -- shards 80-127; a slice no model's training touched.
                     Requires downloading those shards first.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import IterableDataset

CANON_FRAMES = 16
SRC_FPS = 5.0
TGT_FPS = 4.0
CAMERA = "image_0"
IMG_SIZE = 224
ACTION_DIM = 7
STATE_DIM = 7

# floor(15 * 1.25) = 18 -> an episode needs > 19 raw steps to yield one clip.
_LAST_OFFSET = int(np.floor((CANON_FRAMES - 1) * SRC_FPS / TGT_FPS))
_MIN_STEPS = _LAST_OFFSET + 2

VAL_LOCAL = "val[:4295]"     # 80 shards on disk
VAL_FRESH = "val[4295:]"     # shards 80-127, download required

_CANON_OFFSETS = np.floor(np.arange(CANON_FRAMES) * SRC_FPS / TGT_FPS).astype(np.int64)


@dataclass
class CanonicalBatch:
    """What a batch from :func:`collate` looks like (tensors, batch-first).

    frames  : (B, N, H, W, 3) uint8         -- N = CANON_FRAMES
    actions : (B, N, 7) float32             -- actions[:, t] is the raw BridgeData
                                              action at canonical frame t's source step
    states  : (B, N, 7) float32             -- EE state at each canonical frame
    episode_id : (B,) long                  -- index into the split, for episode-safe probes
    """

    frames: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor
    episode_id: torch.Tensor


class CanonicalClips(IterableDataset):
    def __init__(
        self,
        data_root: str = "raw/bridge_dataset/1.0.0",
        split: str = VAL_LOCAL,
        max_clips: int | None = None,
        seed: int = 0,
    ):
        self.data_root = str(data_root)
        self.split = split
        self.max_clips = max_clips
        self.seed = seed

    def __iter__(self):
        import tensorflow as tf
        import tensorflow_datasets as tfds

        tf.config.set_visible_devices([], "GPU")

        info = torch.utils.data.get_worker_info()
        if info is not None and info.num_workers > 1:
            raise RuntimeError("CanonicalClips streams one tfds pipeline; use num_workers=0")

        builder = tfds.builder_from_directory(self.data_root)
        ds = builder.as_dataset(split=self.split, shuffle_files=False)
        rng = random.Random(self.seed)

        n = 0
        for ep_id, ep in enumerate(tfds.as_numpy(ds)):
            steps = list(ep["steps"])
            if len(steps) < _MIN_STEPS:
                continue
            im = np.stack([s["observation"][CAMERA] for s in steps])
            ac = np.stack([s["action"] for s in steps]).astype(np.float32)
            st = np.stack([s["observation"]["state"] for s in steps]).astype(np.float32)

            start = rng.randrange(0, len(im) - _LAST_OFFSET - 1)
            idx = start + _CANON_OFFSETS

            clip = im[idx]                                          # (N, H0, W0, 3) uint8
            if clip.shape[1:3] != (IMG_SIZE, IMG_SIZE):
                # Resize once, here -- every model then sees identical pixels (Level A).
                # bilinear, no antialias: matches both existing pipelines.
                clip = tf.image.resize(clip, [IMG_SIZE, IMG_SIZE], method="bilinear")
                clip = tf.cast(tf.round(tf.clip_by_value(clip, 0.0, 255.0)), tf.uint8).numpy()

            yield {
                "frames": torch.from_numpy(np.ascontiguousarray(clip)),  # (N, 224, 224, 3) uint8
                "actions": torch.from_numpy(ac[idx].copy()),       # (N, 7)
                "states": torch.from_numpy(st[idx].copy()),        # (N, 7)
                "episode_id": ep_id,
            }
            n += 1
            if self.max_clips and n >= self.max_clips:
                return


def collate(items: list[dict]) -> CanonicalBatch:
    return CanonicalBatch(
        frames=torch.stack([it["frames"] for it in items]),
        actions=torch.stack([it["actions"] for it in items]),
        states=torch.stack([it["states"] for it in items]),
        episode_id=torch.tensor([it["episode_id"] for it in items], dtype=torch.long),
    )


def build_loader(
    split: str = VAL_LOCAL,
    max_clips: int | None = None,
    batch_size: int = 8,
    seed: int = 0,
    data_root: str = "raw/bridge_dataset/1.0.0",
):
    from torch.utils.data import DataLoader

    ds = CanonicalClips(data_root=data_root, split=split, max_clips=max_clips, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=collate)
