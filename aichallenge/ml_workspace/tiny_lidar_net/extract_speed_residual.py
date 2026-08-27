#!/usr/bin/env python3
"""Extract synchronized samples for a speed-conditioned steering residual."""

import argparse
import json
from pathlib import Path

import numpy as np

from extract_dagger import read_bag
from extract_data_from_bag import synchronize_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-range", type=float, default=30.0)
    parser.add_argument("--max-sync-ms", type=float, default=150.0)
    args = parser.parse_args()

    streams = read_bag(args.bag.resolve(), args.max_range)
    scans, scan_times = streams["/sensing/lidar/scan"]
    student, student_times = streams["/control/command/control_cmd"]
    teacher, teacher_times = streams["/teacher/control_cmd"]
    speeds, speed_times = streams["/vehicle/status/velocity_status"]
    if any(len(values) == 0 for values in (scans, student, teacher, speeds)):
        raise RuntimeError("one or more required topics contain no messages")

    student_indices, student_deltas = synchronize_data(scan_times, student_times)
    teacher_indices, teacher_deltas = synchronize_data(scan_times, teacher_times)
    speed_indices, speed_deltas = synchronize_data(scan_times, speed_times)
    max_sync_ns = args.max_sync_ms * 1e6
    valid = (
        (student_deltas <= max_sync_ns)
        & (teacher_deltas <= max_sync_ns)
        & (speed_deltas <= max_sync_ns)
    )
    if not np.any(valid):
        raise RuntimeError("no samples met the synchronization limit")

    args.output.mkdir(parents=True, exist_ok=True)
    synced_student = student[student_indices, 0][valid]
    synced_teacher = teacher[teacher_indices, 0][valid]
    synced_speeds = speeds[speed_indices][valid]
    relative_times = (scan_times[valid] - scan_times[valid][0]) / 1e9
    np.save(args.output / "scans.npy", scans[valid])
    np.save(args.output / "student_steers.npy", synced_student)
    np.save(args.output / "teacher_steers.npy", synced_teacher)
    np.save(args.output / "speeds.npy", synced_speeds)
    np.save(args.output / "times_s.npy", relative_times)

    residuals = synced_teacher - synced_student
    report = {
        "samples": int(np.sum(valid)),
        "duration_s": float(relative_times[-1]),
        "speed_mean_mps": float(np.mean(np.abs(synced_speeds))),
        "speed_max_mps": float(np.max(np.abs(synced_speeds))),
        "samples_over_2_5_mps": int(np.sum(np.abs(synced_speeds) > 2.5)),
        "samples_over_3_0_mps": int(np.sum(np.abs(synced_speeds) > 3.0)),
        "residual_mae_rad": float(np.mean(np.abs(residuals))),
        "residual_p95_rad": float(np.percentile(np.abs(residuals), 95)),
        "teacher_sync_mean_ms": float(np.mean(teacher_deltas[valid]) / 1e6),
        "teacher_sync_max_ms": float(np.max(teacher_deltas[valid]) / 1e6),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
