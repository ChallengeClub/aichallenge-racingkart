#!/usr/bin/env python3
"""Record one official E2E-mode controller run without driving the vehicle."""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from autoware_auto_vehicle_msgs.msg import VelocityReport
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class E2EMonitor(Node):
    def __init__(self):
        super().__init__("e2e_evaluation_monitor")
        self.latest_speed = None
        self.max_speed = 0.0
        self.speed_sum = 0.0
        self.speed_samples = 0
        self.max_lap = 0
        self.max_section = 0
        self.latest_lap = 0
        self.latest_section = 0
        self.reported_lap_time_s = 0.0
        self.first_lap_wall_time = None
        self.last_moving_wall_time = None

        self.create_subscription(
            VelocityReport,
            "/vehicle/status/velocity_status",
            self._on_velocity,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/awsim/status",
            self._on_status,
            10,
        )

    def _on_velocity(self, msg):
        speed = max(0.0, float(msg.longitudinal_velocity))
        self.latest_speed = speed
        self.max_speed = max(self.max_speed, speed)
        self.speed_sum += speed
        self.speed_samples += 1
        if speed >= 0.2:
            self.last_moving_wall_time = time.monotonic()

    def _on_status(self, msg):
        if len(msg.data) < 4:
            return
        lap = int(msg.data[1])
        section = int(msg.data[3])
        self.latest_lap = lap
        self.latest_section = section
        self.max_lap = max(self.max_lap, lap)
        self.max_section = max(self.max_section, section)
        self.reported_lap_time_s = float(msg.data[2])
        if self.first_lap_wall_time is None and lap >= 2:
            self.first_lap_wall_time = time.monotonic()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-sec", type=float, default=240.0)
    parser.add_argument("--stopped-timeout-sec", type=float, default=15.0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = E2EMonitor()
    started = time.monotonic()
    stop_reason = "duration_limit"
    last_report = started

    try:
        while time.monotonic() - started < args.duration_sec:
            rclpy.spin_once(node, timeout_sec=0.1)
            now = time.monotonic()
            if now - last_report >= 10.0:
                print(
                    f"elapsed={now - started:.1f}s speed={node.latest_speed or 0.0:.2f} "
                    f"lap={node.latest_lap} section={node.latest_section}",
                    flush=True,
                )
                last_report = now

            moved = node.max_speed >= 1.0 and node.last_moving_wall_time is not None
            if moved and now - node.last_moving_wall_time >= args.stopped_timeout_sec:
                stop_reason = "stopped"
                break
    finally:
        elapsed = time.monotonic() - started
        first_lap_elapsed = (
            node.first_lap_wall_time - started
            if node.first_lap_wall_time is not None
            else None
        )
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_wall_time_s": elapsed,
            "stop_reason": stop_reason,
            "completed_at_least_one_lap": node.max_lap >= 2,
            "first_lap_wall_time_s": first_lap_elapsed,
            "max_lap": node.max_lap,
            "max_section": node.max_section,
            "final_lap": node.latest_lap,
            "final_section": node.latest_section,
            "mean_speed_mps": (
                node.speed_sum / node.speed_samples if node.speed_samples else 0.0
            ),
            "max_speed_mps": node.max_speed,
            "final_speed_mps": node.latest_speed or 0.0,
            "reported_lap_time_s": node.reported_lap_time_s,
        }
        node.destroy_node()
        rclpy.shutdown()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("E2E_RESULT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
