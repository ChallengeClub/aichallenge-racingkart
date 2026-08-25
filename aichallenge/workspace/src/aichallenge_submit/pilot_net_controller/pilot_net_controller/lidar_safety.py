"""Limited LiDAR safety residual for the camera-centred E2E controller."""

import math

import numpy as np


class LidarSafetyController:
    """Slow down and steer toward free space only when the path ahead is blocked."""

    def __init__(
        self,
        *,
        activation_distance_m: float,
        stop_distance_m: float,
        max_steering_correction: float,
        minimum_speed_scale: float = 0.0,
        minimum_steering_correction: float = 0.0,
        recovery_hold_steps: int = 0,
        front_half_angle_deg: float = 15.0,
        side_min_angle_deg: float = 20.0,
        side_max_angle_deg: float = 80.0,
    ) -> None:
        if stop_distance_m < 0.0 or stop_distance_m >= activation_distance_m:
            raise ValueError("stop_distance_m must be below activation_distance_m")
        self.activation_distance_m = float(activation_distance_m)
        self.stop_distance_m = float(stop_distance_m)
        self.max_steering_correction = float(max_steering_correction)
        self.minimum_speed_scale = float(np.clip(minimum_speed_scale, 0.0, 1.0))
        self.minimum_steering_correction = float(
            np.clip(minimum_steering_correction, 0.0, max_steering_correction)
        )
        self.recovery_hold_steps = max(0, int(recovery_hold_steps))
        self._hold_remaining = 0
        self._held_correction = 0.0
        self._held_speed_scale = 1.0
        self.front_half_angle = math.radians(front_half_angle_deg)
        self.side_min_angle = math.radians(side_min_angle_deg)
        self.side_max_angle = math.radians(side_max_angle_deg)

    @staticmethod
    def _clearance(ranges, mask, fallback):
        values = ranges[mask]
        if values.size == 0:
            return fallback
        return float(np.quantile(values, 0.2))

    def compute(
        self,
        ranges,
        angles,
        range_min: float,
        range_max: float,
        force_recovery: bool = False,
        early_activation: bool = False,
    ):
        ranges = np.asarray(ranges, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float32)
        valid = (
            np.isfinite(ranges)
            & (ranges >= float(range_min))
            & (ranges <= float(range_max))
        )
        front = valid & (np.abs(angles) <= self.front_half_angle)
        left = valid & (angles >= self.side_min_angle) & (angles <= self.side_max_angle)
        right = valid & (angles <= -self.side_min_angle) & (angles >= -self.side_max_angle)

        front_clearance = self._clearance(
            ranges, front, self.activation_distance_m
        )
        if (
            front_clearance >= self.activation_distance_m
            and not force_recovery
            and not early_activation
        ):
            if self._hold_remaining > 0:
                self._hold_remaining -= 1
                return (
                    self._held_speed_scale,
                    self._held_correction,
                    front_clearance,
                )
            return 1.0, 0.0, front_clearance

        left_clearance = self._clearance(ranges, left, self.activation_distance_m)
        right_clearance = self._clearance(ranges, right, self.activation_distance_m)
        open_side_ratio = (left_clearance - right_clearance) / max(
            left_clearance + right_clearance, 1e-6
        )
        if abs(open_side_ratio) < 1e-3:
            # During forced recovery, a deterministic fallback is preferable
            # to repeatedly commanding straight into an unseen side contact.
            steering_correction = self._held_correction or (
                self.minimum_steering_correction if force_recovery else 0.0
            )
        else:
            magnitude = max(
                self.minimum_steering_correction,
                self.max_steering_correction * abs(open_side_ratio),
            )
            steering_correction = math.copysign(magnitude, open_side_ratio)
        speed_scale = (
            self.minimum_speed_scale
            if force_recovery
            else (front_clearance - self.stop_distance_m)
            / (self.activation_distance_m - self.stop_distance_m)
        )
        speed_scale = float(np.clip(speed_scale, self.minimum_speed_scale, 1.0))
        self._hold_remaining = self.recovery_hold_steps
        self._held_correction = float(steering_correction)
        self._held_speed_scale = speed_scale
        return speed_scale, float(steering_correction), front_clearance
