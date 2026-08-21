"""Download the bounded BridgeData V2 train and validation subsets.

Usage::

    python bridge_data/download.py --out raw/bridge_dataset/1.0.0

Downloads:
    - first 400 official train shards
    - first 80 official validation shards

The downloader is resumable at the FILE level, but only a fully verified
TFRecord is promoted to its final filename.

Incomplete/corrupted files are never treated as valid downloads.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import tensorflow as tf


SOURCE = (
    "https://rail.eecs.berkeley.edu/"
    "datasets/bridge_release/data/tfds/bridge_dataset/1.0.0"
)

# ---------------------------------------------------------
# YOUR BOUNDED SUBSET
# ---------------------------------------------------------

TRAIN_SHARDS = 400
VAL_SHARDS = 80

METADATA = (
    "dataset_info.json",
    "features.json",
)

# Number of download attempts for a file.
MAX_RETRIES = 5

# Seconds between retries.
RETRY_DELAY = 5

# Download block size.
BLOCK_SIZE = 1024 * 1024


def shard_names(split: str, count: int) -> list[str]:
    """Return only the requested subset of official shard names."""

    total = 1024 if split == "train" else 128

    if count < 1 or count > total:
        raise ValueError(
            f"{split}: count must be between 1 and {total}, got {count}"
        )

    return [
        f"bridge_dataset-{split}.tfrecord-{i:05d}-of-{total:05d}"
        for i in range(count)
    ]


def verify_tfrecord(path: Path) -> tuple[bool, int, str | None]:
    """Read the complete TFRecord and verify that it is not truncated.

    Returns:
        (is_valid, number_of_records, error_message)
    """

    try:
        count = 0

        dataset = tf.data.TFRecordDataset(
            str(path),
            num_parallel_reads=1,
        )

        for _ in dataset:
            count += 1

        return True, count, None

    except Exception as exc:
        return False, count, str(exc)


def download_once(name: str, target: Path) -> int:
    """Download one file to a .partial file.

    The final target is NEVER written directly.
    """

    partial = target.with_name(target.name + ".partial")

    # Remove an old partial from a previous failed attempt.
    if partial.exists():
        partial.unlink()

    url = f"{SOURCE}/{name}"

    print(f"[DOWNLOAD] {name}", flush=True)

    try:
        with urlopen(url, timeout=120) as response:
            with partial.open("wb") as stream:

                while True:
                    block = response.read(BLOCK_SIZE)

                    if not block:
                        break

                    stream.write(block)

                stream.flush()
                os.fsync(stream.fileno())

        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError(
                f"Downloaded file is empty: {partial}"
            )

        return partial.stat().st_size

    except Exception:
        # Never leave a possibly incomplete .partial file around.
        if partial.exists():
            partial.unlink()

        raise


def download_file(name: str, out: Path) -> dict[str, object]:
    """Download and VERIFY one file.

    Existing final TFRecords are verified before being reused.

    If verification fails, the existing file is deleted and downloaded again.
    """

    target = out / name

    target.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # METADATA
    # -----------------------------------------------------

    if name in METADATA:

        if target.exists() and target.stat().st_size > 0:
            return {
                "name": name,
                "bytes": target.stat().st_size,
                "reused": True,
                "verified": True,
                "records": None,
            }

        for attempt in range(1, MAX_RETRIES + 1):

            try:
                size = download_once(name, target)

                partial = target.with_name(target.name + ".partial")

                os.replace(partial, target)

                return {
                    "name": name,
                    "bytes": size,
                    "reused": False,
                    "verified": True,
                    "records": None,
                }

            except Exception as exc:

                print(
                    f"[RETRY {attempt}/{MAX_RETRIES}] "
                    f"{name}: {exc}",
                    flush=True,
                )

                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        raise RuntimeError(
            f"Failed to download {name} after {MAX_RETRIES} attempts"
        )

    # -----------------------------------------------------
    # TFRECORD
    # -----------------------------------------------------

    if target.exists() and target.stat().st_size > 0:

        valid, records, error = verify_tfrecord(target)

        if valid:
            print(
                f"[REUSE] {name} "
                f"records={records} "
                f"bytes={target.stat().st_size}",
                flush=True,
            )

            return {
                "name": name,
                "bytes": target.stat().st_size,
                "reused": True,
                "verified": True,
                "records": records,
            }

        # Existing file is corrupted.
        print(
            f"[CORRUPTED] {name}",
            flush=True,
        )
        print(
            f"             {error}",
            flush=True,
        )
        print(
            f"[DELETE]    {name}",
            flush=True,
        )

        target.unlink()

    # -----------------------------------------------------
    # DOWNLOAD + VERIFY + ATOMIC RENAME
    # -----------------------------------------------------

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            size = download_once(name, target)

            partial = target.with_name(target.name + ".partial")

            print(
                f"[VERIFY] {name}",
                flush=True,
            )

            valid, records, error = verify_tfrecord(partial)

            if not valid:
                print(
                    f"[BAD DOWNLOAD] {name}",
                    flush=True,
                )
                print(
                    f"                {error}",
                    flush=True,
                )

                if partial.exists():
                    partial.unlink()

                raise RuntimeError(
                    "Downloaded TFRecord failed verification"
                )

            # -------------------------------------------------
            # CRITICAL:
            #
            # The final filename is created ONLY after
            # TensorFlow successfully reads the entire file.
            # -------------------------------------------------

            os.replace(partial, target)

            print(
                f"[OK] {name} "
                f"records={records} "
                f"bytes={size}",
                flush=True,
            )

            return {
                "name": name,
                "bytes": size,
                "reused": False,
                "verified": True,
                "records": records,
            }

        except Exception as exc:

            print(
                f"[RETRY {attempt}/{MAX_RETRIES}] "
                f"{name}: {exc}",
                flush=True,
            )

            partial = target.with_name(target.name + ".partial")

            if partial.exists():
                partial.unlink()

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    raise RuntimeError(
        f"Failed to download and verify {name} "
        f"after {MAX_RETRIES} attempts"
    )


def download_all(
    out: str | Path = "raw/bridge_dataset/1.0.0",
    workers: int = 8,
) -> dict:
    """Download and verify the complete bounded subset."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    names = (
        list(METADATA)
        + shard_names("train", TRAIN_SHARDS)
        + shard_names("val", VAL_SHARDS)
    )

    print("=" * 70)
    print("BridgeData V2 bounded download")
    print("=" * 70)
    print(f"Train shards : {TRAIN_SHARDS}")
    print(f"Val shards   : {VAL_SHARDS}")
    print(f"Total files  : {len(names)}")
    print(f"Workers      : {workers}")
    print("=" * 70)

    results: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:

        futures = [
            pool.submit(download_file, name, out)
            for name in names
        ]

        for index, future in enumerate(
            as_completed(futures), 1
        ):

            # IMPORTANT:
            # If ANY shard cannot be downloaded/verified,
            # abort the entire download instead of pretending
            # the dataset is complete.
            result = future.result()

            results.append(result)

            print(
                f"[PROGRESS] {index}/{len(futures)} files complete",
                flush=True,
            )

    # ---------------------------------------------------------
    # FINAL MANIFEST
    # ---------------------------------------------------------

    manifest = {
        "dataset": "BridgeData V2 official TFDS/RLDS release",
        "source": SOURCE,
        "data_root": str(out),
        "train_shards": TRAIN_SHARDS,
        "val_shards": VAL_SHARDS,
        "objects": sorted(
            results,
            key=lambda item: str(item["name"]),
        ),
    }

    manifest_path = out.parent / "manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print(f"Manifest: {manifest_path}")
    print("=" * 70)

    return manifest


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("raw/bridge_dataset/1.0.0"),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    download_all(
        args.out,
        args.workers,
    )


if __name__ == "__main__":
    main()