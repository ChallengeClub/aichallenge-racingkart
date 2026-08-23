#!/usr/bin/env python3
"""Create leakage-resistant train/validation views from one extracted sequence."""

import argparse
import json
from pathlib import Path

import numpy as np


FILES = ("scans.npy", "steers.npy", "accelerations.npy")


def save_slice(source: Path, destination: Path, selection: slice) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    lengths = set()
    for filename in FILES:
        values = np.load(source / filename, mmap_mode="r")
        selected = np.asarray(values[selection])
        np.save(destination / filename, selected)
        lengths.add(len(selected))
    if len(lengths) != 1:
        raise ValueError(f"inconsistent source lengths in {source}")
    return lengths.pop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--gap-fraction", type=float, default=0.05)
    args = parser.parse_args()

    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train-fraction must be between zero and one")
    if args.gap_fraction < 0.0 or args.train_fraction + args.gap_fraction >= 1.0:
        raise ValueError("invalid gap-fraction")

    total = len(np.load(args.source / "scans.npy", mmap_mode="r"))
    train_end = int(total * args.train_fraction)
    validation_start = int(total * (args.train_fraction + args.gap_fraction))
    train_count = save_slice(
        args.source,
        args.output / "train" / "run01_early",
        slice(0, train_end),
    )
    validation_count = save_slice(
        args.source,
        args.output / "val" / "run01_late",
        slice(validation_start, total),
    )
    report = {
        "total_samples": total,
        "train_samples": train_count,
        "gap_samples": validation_start - train_end,
        "validation_samples": validation_count,
        "train_fraction": args.train_fraction,
        "gap_fraction": args.gap_fraction,
    }
    (args.output / "split.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
