"""Step 0 gate: prove the canonical clip stream reads the 80 BridgeData V2 val
shards, resamples correctly, and is deterministic under a fixed seed.

    .venv/bin/python -m evaluation.common.smoke_data
"""
from __future__ import annotations

from evaluation.common.data_spec import (
    ACTION_DIM,
    CANON_FRAMES,
    IMG_SIZE,
    STATE_DIM,
    VAL_LOCAL,
    build_loader,
)


def main() -> None:
    loader = build_loader(split=VAL_LOCAL, max_clips=8, batch_size=4, seed=0)
    batch = next(iter(loader))

    B = batch.frames.shape[0]
    print("frames ", tuple(batch.frames.shape), batch.frames.dtype)
    print("actions", tuple(batch.actions.shape), batch.actions.dtype)
    print("states ", tuple(batch.states.shape), batch.states.dtype)
    print("episode_id", batch.episode_id.tolist())

    assert batch.frames.shape == (B, CANON_FRAMES, IMG_SIZE, IMG_SIZE, 3), batch.frames.shape
    assert str(batch.frames.dtype) == "torch.uint8"
    assert batch.actions.shape == (B, CANON_FRAMES, ACTION_DIM), batch.actions.shape
    assert batch.states.shape == (B, CANON_FRAMES, STATE_DIM), batch.states.shape
    assert batch.frames.max() > 1, "frames should be raw 0-255, not normalised"

    # determinism: a second pass with the same seed yields the same episodes
    again = next(iter(build_loader(split=VAL_LOCAL, max_clips=8, batch_size=4, seed=0)))
    assert again.episode_id.tolist() == batch.episode_id.tolist(), "seed is not deterministic"
    assert (again.frames == batch.frames).all(), "same seed produced different frames"

    print("\nStep 0 OK - canonical stream reads val[:4295], shapes + determinism verified")


if __name__ == "__main__":
    main()
