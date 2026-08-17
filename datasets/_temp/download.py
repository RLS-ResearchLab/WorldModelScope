"""Download the bounded BridgeData V2 train and validation subsets.

Usage::

    python bridge_data/download.py --out raw/bridge_dataset/1.0.0

The subset is the first 400 official train shards and first 80 official
validation shards (about 46 and 8.6 hours respectively). Downloads are
resumable at the file level and are written atomically via ``.partial`` files.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

SOURCE = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0"
TRAIN_SHARDS = 400
VAL_SHARDS = 80
METADATA = ("dataset_info.json", "features.json")


def shard_names(split: str, count: int) -> list[str]:
    total = 1024 if split == "train" else 128
    return [f"bridge_dataset-{split}.tfrecord-{i:05d}-of-{total:05d}" for i in range(count)]


def download_file(name: str, out: Path) -> dict[str, object]:
    target = out / name
    if target.exists() and target.stat().st_size > 0:
        return {"name": name, "bytes": target.stat().st_size, "reused": True}

    partial = target.with_name(target.name + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(f"{SOURCE}/{name}") as response, partial.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    os.replace(partial, target)
    return {"name": name, "bytes": target.stat().st_size, "reused": False}


def download_all(out: str | Path = "raw/bridge_dataset/1.0.0", workers: int = 8) -> dict:
    """Download and return a manifest for the complete bounded subset."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = list(METADATA) + shard_names("train", TRAIN_SHARDS) + shard_names("val", VAL_SHARDS)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(download_file, name, out) for name in names]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if index == len(futures) or index % 20 == 0:
                print(f"{index}/{len(futures)} files complete", flush=True)

    manifest = {
        "dataset": "BridgeData V2 official TFDS/RLDS release",
        "source": SOURCE,
        "data_root": str(out),
        "train_shards": TRAIN_SHARDS,
        "val_shards": VAL_SHARDS,
        "objects": sorted(results, key=lambda item: str(item["name"])),
    }
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("raw/bridge_dataset/1.0.0"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    download_all(args.out, args.workers)


if __name__ == "__main__":
    main()
