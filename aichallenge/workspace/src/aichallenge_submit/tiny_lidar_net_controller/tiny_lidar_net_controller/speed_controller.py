"""Longitudinal controller for the TinyLiDARNet submission baseline."""

import math

import numpy as np


class SpeedController:
    """Convert wheel-speed error into a bounded acceleration command."""

    def __init__(
        self,
        *,
        target_speed_mps: float,
        proportional_gain: float,
        min_acceleration: float,
        max_acceleration: float,
    ) -> None:
        if target_speed_mps < 0.0:
            raise ValueError("target_speed_mps must be non-negative")
        if proportional_gain < 0.0:
            raise ValueError("proportional_gain must be non-negative")
        if min_acceleration > max_acceleration:
            raise ValueError("min_acceleration must not exceed max_acceleration")
        self.target_speed_mps = float(target_speed_mps)
        self.proportional_gain = float(proportional_gain)
        self.min_acceleration = float(min_acceleration)
        self.max_acceleration = float(max_acceleration)

    def compute(
        self,
        current_speed_mps: float,
        target_speed_mps: float | None = None,
    ) -> float:
        speed = max(0.0, float(current_speed_mps))
        target = (
            self.target_speed_mps
            if target_speed_mps is None
            else max(0.0, float(target_speed_mps))
        )
        acceleration = self.proportional_gain * (
            target - speed
        )
        return max(
            self.min_acceleration,
            min(self.max_acceleration, acceleration),
        )


def select_target_speed(
    *,
    base_speed_mps: float,
    straight_speed_mps: float,
    steering_angle: float,
    max_straight_steering: float,
    front_clearance_m: float,
    minimum_straight_clearance_m: float,
    safety_speed_scale: float,
) -> float:
    """Raise speed only on a clear straight; safety scaling always wins."""
    safety_scale = max(0.0, min(1.0, float(safety_speed_scale)))
    is_clear_straight = (
        safety_scale >= 1.0
        and abs(float(steering_angle)) <= float(max_straight_steering)
        and float(front_clearance_m) >= float(minimum_straight_clearance_m)
    )
    cruise_speed = straight_speed_mps if is_clear_straight else base_speed_mps
    return max(0.0, float(cruise_speed) * safety_scale)


def calculate_forward_clearance(
    ranges,
    angles,
    *,
    range_min: float,
    range_max: float,
    half_angle_deg: float = 10.0,
) -> float:
    """Return a robust forward clearance, treating no return as open space."""
    ranges = np.asarray(ranges, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)
    usable = np.where(np.isposinf(ranges), float(range_max), ranges)
    valid = (
        np.isfinite(usable)
        & (usable >= float(range_min))
        & (usable <= float(range_max))
        & (np.abs(angles) <= math.radians(half_angle_deg))
    )
    values = usable[valid]
    if values.size == 0:
        return 0.0
    return float(np.quantile(values, 0.2))
