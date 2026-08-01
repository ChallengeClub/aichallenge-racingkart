"""Create reproducible sequence-level train/validation splits and quality metadata."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np


REQUIRED = ("images.npy", "steers.npy", "accelerations.npy")


def summarize(sequence: Path) -> dict[str, object]:
    images = np.load(sequence / "images.npy", mmap_mode="r")
    steers = np.load(sequence / "steers.npy")
    accelerations = np.load(sequence / "accelerations.npy")
    if not (len(images) == len(steers) == len(accelerations)):
        raise ValueError(f"length mismatch in {sequence.name}")
    deltas_path = sequence / "delta_times.npy"
    deltas = np.load(deltas_path) if deltas_path.exists() else np.array([])
    return {
        "sequence": sequence.name,
        "samples": int(len(images)),
        "image_shape": list(images.shape[1:]),
        "steering_min": float(steers.min()),
        "steering_max": float(steers.max()),
        "steering_mean": float(steers.mean()),
        "acceleration_min": float(accelerations.min()),
        "acceleration_max": float(accelerations.max()),
        "acceleration_mean": float(accelerations.mean()),
        "sync_delta_mean_s": float(deltas.mean()) if len(deltas) else None,
        "sync_delta_max_s": float(deltas.max()) if len(deltas) else None,
    }


def materialize_training_view(view: Path, source: Path) -> None:
    """Create a compact training view while preserving raw extracted labels."""
    if view.is_symlink():
        view.unlink()
    elif view.exists():
        shutil.rmtree(view)
    view.mkdir()
    for name in ("images.npy", "steers.npy"):
        target = source / name
        (view / name).symlink_to(os.path.relpath(target, view), target_is_directory=False)
    accelerations = np.clip(np.load(source / "accelerations.npy"), -1.0, 1.0)
    np.save(view / "accelerations.npy", accelerations)


def main(collection: Path, val_fraction: float) -> None:
    extracted = collection / "extracted"
    rejected: dict[str, str] = {}
    quality_path = collection / "quality_report.json"
    if quality_path.exists():
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        for item in quality.get("sequences", []):
            stopped = item.get("stopped_fraction_below_0_2_mps")
            if stopped is not None and float(stopped) > 0.2:
                rejected[str(item["sequence"])] = f"stopped_fraction={float(stopped):.3f}"
    sequences = sorted(
        p for p in extracted.iterdir()
        if p.is_dir()
        and p.name not in rejected
        and all((p / name).exists() for name in REQUIRED)
    )
    if len(sequences) < 2:
        raise RuntimeError("At least two valid sequences are required for a sequence-level split")

    # Split independently within each speed group so every collected speed is
    # represented in both train and validation when repeated runs are available.
    groups: dict[str, list[Path]] = {}
    for sequence in sequences:
        match = re.match(r"^(v[0-9]+kmh)_run[0-9]+$", sequence.name)
        group = match.group(1) if match else "ungrouped"
        groups.setdefault(group, []).append(sequence)
    val_names: set[str] = set()
    for group_sequences in groups.values():
        if len(group_sequences) < 2:
            continue
        val_count = max(1, round(len(group_sequences) * val_fraction))
        val_names.update(p.name for p in group_sequences[-val_count:])
    if not val_names:
        val_names.add(sequences[-1].name)
    summaries = []
    for split in ("train", "val"):
        split_dir = collection / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir()
    for sequence in sequences:
        split = "val" if sequence.name in val_names else "train"
        materialize_training_view(collection / split / sequence.name, sequence)
        item = summarize(sequence)
        item["split"] = split
        summaries.append(item)

    manifest = {
        "schema_version": 1,
        "split_unit": "sequence",
        "training_label_transform": {"acceleration_clip": [-1.0, 1.0]},
        "val_fraction": val_fraction,
        "sequences": summaries,
        "rejected_sequences": rejected,
        "total_samples": sum(int(item["samples"]) for item in summaries),
    }
    (collection / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.34)
    args = parser.parse_args()
    main(args.collection.resolve(), args.val_fraction)
