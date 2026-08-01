from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces

from action.interfaces import ActionAdapter
from context.context_types import StepContext


class RacelineResidualActionAdapter(ActionAdapter):
    """Pure-pursuit/throttle baseline with bounded SAC residual actions."""

    def __init__(
        self,
        raceline_csv_path: str,
        lookahead_index: int = 5,
        wheelbase_m: float = 1.0,
        target_speed_mps: float = 4.0,
        speed_kp: float = 0.4,
        steering_residual_scale: float = 0.2,
        acceleration_residual_scale: float = 0.2,
        max_steering: float = 1.0,
        steering_gain: float = 1.0,
        loop_start_index: int = 0,
    ) -> None:
        path = Path(raceline_csv_path)
        if not path.is_file():
            raise ValueError(f"Raceline CSV not found: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self._raceline = np.asarray(
            [[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float64
        )
        if len(self._raceline) < 2:
            raise ValueError("Raceline requires at least two points")
        self.lookahead_index = int(lookahead_index)
        self.wheelbase_m = float(wheelbase_m)
        self.target_speed_mps = float(target_speed_mps)
        self.speed_kp = float(speed_kp)
        self.steering_residual_scale = float(steering_residual_scale)
        self.acceleration_residual_scale = float(acceleration_residual_scale)
        self.max_steering = float(max_steering)
        self.steering_gain = float(steering_gain)
        self.loop_start_index = int(loop_start_index)
        self._nearest_index: int | None = None

    def reset(self) -> None:
        self._nearest_index = None

    def _wrap_index(self, index: int) -> int:
        if self.loop_start_index <= index < len(self._raceline):
            return index
        loop_length = len(self._raceline) - self.loop_start_index
        return self.loop_start_index + (index - self.loop_start_index) % loop_length

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    @staticmethod
    def _value(context: StepContext | None, name: str, default: float) -> float:
        if context is None:
            return default
        return float(context.env_state.get_value(name, default))

    def adapt(self, action: np.ndarray, context: StepContext | None = None) -> dict[str, float]:
        x = self._value(context, "kinematic_pose_x_m", float("nan"))
        y = self._value(context, "kinematic_pose_y_m", float("nan"))
        speed = max(0.0, self._value(context, "vehicle_speed_mps", 0.0))

        baseline_steering = 0.0
        if math.isfinite(x) and math.isfinite(y):
            qx = self._value(context, "kinematic_orientation_x", 0.0)
            qy = self._value(context, "kinematic_orientation_y", 0.0)
            qz = self._value(context, "kinematic_orientation_z", 0.0)
            qw = self._value(context, "kinematic_orientation_w", 1.0)
            yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

            position = np.asarray([x, y], dtype=np.float64)
            if self._nearest_index is None:
                nearest = int(np.argmin(np.sum((self._raceline - position) ** 2, axis=1)))
            else:
                candidate_indices = [
                    self._wrap_index(self._nearest_index + offset) for offset in range(-3, 11)
                ]
                nearest = min(
                    candidate_indices,
                    key=lambda index: float(np.sum((self._raceline[index] - position) ** 2)),
                )
            self._nearest_index = nearest
            target_index = self._wrap_index(nearest + self.lookahead_index)
            target = self._raceline[target_index] - position
            target_left = -math.sin(yaw) * target[0] + math.cos(yaw) * target[1]
            distance_squared = max(float(np.dot(target, target)), 1e-6)
            curvature = 2.0 * target_left / distance_squared
            baseline_steering = self.steering_gain * math.atan(self.wheelbase_m * curvature)

        baseline_acceleration = np.clip(
            self.speed_kp * (self.target_speed_mps - speed), 0.0, 1.0
        )
        steering = baseline_steering + self.steering_residual_scale * float(action[0])
        acceleration = baseline_acceleration + self.acceleration_residual_scale * float(action[1])
        return {
            "steering": float(np.clip(steering, -self.max_steering, self.max_steering)),
            "acceleration": float(np.clip(acceleration, 0.0, 1.0)),
        }
