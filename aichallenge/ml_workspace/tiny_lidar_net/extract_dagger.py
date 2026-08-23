#!/usr/bin/env python3
"""Extract high-disagreement and pre-failure samples from a shadow-teacher bag."""

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from extract_data_from_bag import clean_scan_array, synchronize_data


def sustained_stop_time(times: np.ndarray, speeds: np.ndarray, hold_sec: float) -> int | None:
    has_moved = False
    low_start = None
    for timestamp, speed in zip(times, np.abs(speeds)):
        if speed >= 0.8:
            has_moved = True
            low_start = None
        elif has_moved and speed <= 0.2:
            if low_start is None:
                low_start = int(timestamp)
            elif timestamp - low_start >= hold_sec * 1e9:
                return low_start
        else:
            low_start = None
    return None


def read_bag(bag: Path, max_range: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    topics = {
        "/sensing/lidar/scan": ([], []),
        "/control/command/control_cmd": ([], []),
        "/teacher/control_cmd": ([], []),
        "/vehicle/status/velocity_status": ([], []),
    }
    with AnyReader([bag]) as reader:
        connections = [connection for connection in reader.connections if connection.topic in topics]
        for connection, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, connection.msgtype)
            values, times = topics[connection.topic]
            if connection.topic == "/sensing/lidar/scan":
                values.append(clean_scan_array(np.asarray(msg.ranges), max_range))
            elif connection.topic == "/vehicle/status/velocity_status":
                values.append(float(msg.longitudinal_velocity))
            else:
                values.append(
                    [
                        float(msg.lateral.steering_tire_angle),
                        float(msg.longitudinal.acceleration),
                    ]
                )
            times.append(timestamp)
    return {
        topic: (np.asarray(values), np.asarray(times, dtype=np.int64))
        for topic, (values, times) in topics.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disagreement-rad", type=float, default=0.15)
    parser.add_argument("--pre-failure-sec", type=float, default=20.0)
    parser.add_argument("--post-failure-sec", type=float, default=5.0)
    parser.add_argument("--stop-hold-sec", type=float, default=2.0)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-range", type=float, default=30.0)
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
    speed_indices, _ = synchronize_data(scan_times, speed_times)
    synced_student = student[student_indices]
    synced_teacher = teacher[teacher_indices]
    synced_speeds = speeds[speed_indices]
    failure_time = sustained_stop_time(scan_times, synced_speeds, args.stop_hold_sec)
    if failure_time is None:
        raise RuntimeError("no sustained stop found in DAgger bag")

    disagreement = np.abs(synced_teacher[:, 0] - synced_student[:, 0])
    relative_sec = (scan_times - failure_time) / 1e9
    failure_window = (
        (relative_sec >= -args.pre_failure_sec)
        & (relative_sec <= args.post_failure_sec)
    )
    before_post_limit = relative_sec <= args.post_failure_sec
    selected = failure_window | (
        (disagreement >= args.disagreement_rad) & before_post_limit
    )
    selected_indices = np.flatnonzero(selected)
    if len(selected_indices) == 0:
        raise RuntimeError("selection produced no DAgger samples")
    repeated = np.tile(selected_indices, max(1, args.repeat))

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "scans.npy", scans[repeated])
    np.save(args.output / "steers.npy", synced_teacher[repeated, 0])
    np.save(args.output / "accelerations.npy", synced_teacher[repeated, 1])
    np.save(args.output / "student_steers.npy", synced_student[repeated, 0])
    np.save(args.output / "relative_failure_time_s.npy", relative_sec[repeated])
    report = {
        "source_scan_samples": len(scans),
        "failure_time_from_bag_start_s": float((failure_time - scan_times[0]) / 1e9),
        "selected_unique_samples": len(selected_indices),
        "training_samples_after_repeat": len(repeated),
        "repeat": max(1, args.repeat),
        "teacher_student_steering_mae_all": float(np.mean(disagreement)),
        "teacher_student_steering_mae_selected": float(np.mean(disagreement[selected_indices])),
        "selected_pre_failure_samples": int(np.sum(relative_sec[selected_indices] < 0.0)),
        "selected_post_failure_samples": int(np.sum(relative_sec[selected_indices] >= 0.0)),
        "student_sync_delta_mean_ms": float(np.mean(student_deltas) / 1e6),
        "teacher_sync_delta_mean_ms": float(np.mean(teacher_deltas) / 1e6),
        "teacher_sync_delta_max_ms": float(np.max(teacher_deltas) / 1e6),
    }
    (args.output / "dagger_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
