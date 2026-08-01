"""Summarize raw SW-teacher bags without exposing or copying their contents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


def analyze_bag(bag: Path) -> dict[str, object]:
    counts: dict[str, int] = {}
    velocities: list[float] = []
    steering: list[float] = []
    first_stamp: int | None = None
    last_stamp: int | None = None
    with AnyReader([bag]) as reader:
        for connection, timestamp, raw in reader.messages():
            counts[connection.topic] = counts.get(connection.topic, 0) + 1
            first_stamp = timestamp if first_stamp is None else min(first_stamp, timestamp)
            last_stamp = timestamp if last_stamp is None else max(last_stamp, timestamp)
            if connection.topic == "/vehicle/status/velocity_status":
                msg = reader.deserialize(raw, connection.msgtype)
                velocities.append(float(msg.longitudinal_velocity))
            elif connection.topic == "/vehicle/status/steering_status":
                msg = reader.deserialize(raw, connection.msgtype)
                steering.append(float(msg.steering_tire_angle))

    speed = np.abs(np.asarray(velocities, dtype=np.float64))
    steer = np.asarray(steering, dtype=np.float64)
    target_match = re.match(r"^v([0-9]+)kmh_", bag.name)
    return {
        "sequence": bag.name,
        "target_speed_kmh": float(target_match.group(1)) if target_match else None,
        "duration_s": (last_stamp - first_stamp) / 1e9 if first_stamp is not None else 0.0,
        "topic_counts": counts,
        "actual_speed_mean_mps": float(speed.mean()) if len(speed) else None,
        "actual_speed_p95_mps": float(np.percentile(speed, 95)) if len(speed) else None,
        "actual_speed_max_mps": float(speed.max()) if len(speed) else None,
        "stopped_fraction_below_0_2_mps": float(np.mean(speed < 0.2)) if len(speed) else None,
        "actual_steer_min_rad": float(steer.min()) if len(steer) else None,
        "actual_steer_max_rad": float(steer.max()) if len(steer) else None,
    }


def main(collection: Path) -> None:
    bags = sorted(p.parent for p in (collection / "raw").rglob("metadata.yaml"))
    report = {
        "schema_version": 1,
        "collection": collection.name,
        "sequences": [analyze_bag(bag) for bag in bags],
    }
    output = collection / "quality_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    args = parser.parse_args()
    main(args.collection.resolve())
