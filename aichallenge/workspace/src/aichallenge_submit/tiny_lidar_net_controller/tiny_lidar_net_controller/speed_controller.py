"""Longitudinal controller for the TinyLiDARNet submission baseline."""

import math
from collections import deque

import numpy as np


class TimeToCollisionGovernor:
    """Reduce the speed target when forward clearance is closing too quickly.

    A short median history rejects single-scan LiDAR discontinuities.  The
    governor only changes longitudinal speed; the existing safety residual
    remains responsible for last-resort steering close to an obstacle.
    """

    def __init__(
        self,
        *,
        activation_ttc_sec: float,
        minimum_ttc_sec: float,
        minimum_speed_scale: float,
        minimum_closing_speed_mps: float = 0.5,
        rate_history_size: int = 3,
        hold_steps: int = 5,
        maximum_sample_interval_sec: float = 0.5,
    ) -> None:
        if minimum_ttc_sec <= 0.0 or minimum_ttc_sec >= activation_ttc_sec:
            raise ValueError("minimum_ttc_sec must be positive and below activation_ttc_sec")
        if rate_history_size < 1:
            raise ValueError("rate_history_size must be positive")
        self.activation_ttc_sec = float(activation_ttc_sec)
        self.minimum_ttc_sec = float(minimum_ttc_sec)
        self.minimum_speed_scale = float(np.clip(minimum_speed_scale, 0.0, 1.0))
        self.minimum_closing_speed_mps = max(0.0, float(minimum_closing_speed_mps))
        self.maximum_sample_interval_sec = float(maximum_sample_interval_sec)
        self.rate_history_size = int(rate_history_size)
        self.hold_steps = max(0, int(hold_steps))
        self._closing_rates = deque(maxlen=self.rate_history_size)
        self._previous_clearance = None
        self._previous_time = None
        self._hold_remaining = 0
        self._held_scale = 1.0
        self.last_ttc_sec = math.inf
        self.sample_count = 0
        self.intervention_count = 0

    @property
    def intervention_ratio(self) -> float:
        if self.sample_count == 0:
            return 0.0
        return self.intervention_count / self.sample_count

    def reset(self) -> None:
        self._closing_rates.clear()
        self._previous_clearance = None
        self._previous_time = None
        self._hold_remaining = 0
        self._held_scale = 1.0
        self.last_ttc_sec = math.inf

    def compute(self, front_clearance_m: float, timestamp_sec: float) -> float:
        """Return a [minimum_speed_scale, 1] multiplier for the speed target."""
        clearance = max(0.0, float(front_clearance_m))
        now = float(timestamp_sec)
        self.sample_count += 1

        if self._previous_time is not None:
            dt = now - self._previous_time
            if dt <= 0.0 or dt > self.maximum_sample_interval_sec:
                self.reset()
            else:
                closing_rate = max(
                    0.0,
                    (self._previous_clearance - clearance) / dt,
                )
                self._closing_rates.append(closing_rate)

        self._previous_clearance = clearance
        self._previous_time = now

        # Wait for a complete window so a single range jump cannot brake the car.
        if len(self._closing_rates) < self.rate_history_size:
            scale = 1.0
            self.last_ttc_sec = math.inf
        else:
            closing_rate = float(np.median(self._closing_rates))
            if closing_rate < self.minimum_closing_speed_mps:
                scale = 1.0
                self.last_ttc_sec = math.inf
            else:
                self.last_ttc_sec = clearance / max(closing_rate, 1e-6)
                scale = (self.last_ttc_sec - self.minimum_ttc_sec) / (
                    self.activation_ttc_sec - self.minimum_ttc_sec
                )
                scale = float(np.clip(scale, self.minimum_speed_scale, 1.0))

        if scale < 1.0:
            self._held_scale = scale
            self._hold_remaining = self.hold_steps
        elif self._hold_remaining > 0:
            self._hold_remaining -= 1
            scale = self._held_scale
        else:
            self._held_scale = 1.0

        if scale < 1.0:
            self.intervention_count += 1
        return scale


class StuckDetector:
    """Activate recovery only after a moving vehicle remains nearly stopped."""

    def __init__(
        self,
        *,
        stopped_speed_mps: float = 0.2,
        moving_speed_mps: float = 0.8,
        trigger_duration_sec: float = 2.0,
    ) -> None:
        if stopped_speed_mps < 0.0 or moving_speed_mps <= stopped_speed_mps:
            raise ValueError("moving_speed_mps must exceed stopped_speed_mps")
        if trigger_duration_sec < 0.0:
            raise ValueError("trigger_duration_sec must be non-negative")
        self.stopped_speed_mps = float(stopped_speed_mps)
        self.moving_speed_mps = float(moving_speed_mps)
        self.trigger_duration_sec = float(trigger_duration_sec)
        self._has_moved = False
        self._stopped_since = None

    def compute(self, speed_mps: float | None, timestamp_sec: float) -> bool:
        if speed_mps is None:
            return False
        speed = abs(float(speed_mps))
        now = float(timestamp_sec)
        if speed >= self.moving_speed_mps:
            self._has_moved = True
            self._stopped_since = None
            return False
        if not self._has_moved or speed > self.stopped_speed_mps:
            self._stopped_since = None
            return False
        if self._stopped_since is None:
            self._stopped_since = now
            return False
        return now - self._stopped_since >= self.trigger_duration_sec


class SpeedController:
    """Convert wheel-speed error into a bounded acceleration command."""

    def __init__(
        self,
        *,
        target_speed_mps: float,
        proportional_gain: float,
        min_acceleration: float,
        max_acceleration: float,
        hard_braking_threshold_mps: float | None = None,
        hard_min_acceleration: float | None = None,
    ) -> None:
        if target_speed_mps < 0.0:
            raise ValueError("target_speed_mps must be non-negative")
        if proportional_gain < 0.0:
            raise ValueError("proportional_gain must be non-negative")
        if min_acceleration > max_acceleration:
            raise ValueError("min_acceleration must not exceed max_acceleration")
        if hard_braking_threshold_mps is not None and hard_braking_threshold_mps < 0.0:
            raise ValueError("hard_braking_threshold_mps must be non-negative")
        if hard_min_acceleration is not None and hard_min_acceleration > min_acceleration:
            raise ValueError("hard_min_acceleration must not exceed min_acceleration")
        self.target_speed_mps = float(target_speed_mps)
        self.proportional_gain = float(proportional_gain)
        self.min_acceleration = float(min_acceleration)
        self.max_acceleration = float(max_acceleration)
        self.hard_braking_threshold_mps = (
            None
            if hard_braking_threshold_mps is None
            else float(hard_braking_threshold_mps)
        )
        self.hard_min_acceleration = (
            self.min_acceleration
            if hard_min_acceleration is None
            else float(hard_min_acceleration)
        )

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
        lower_bound = self.min_acceleration
        if (
            self.hard_braking_threshold_mps is not None
            and speed - target >= self.hard_braking_threshold_mps
        ):
            lower_bound = self.hard_min_acceleration
        return max(
            lower_bound,
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
