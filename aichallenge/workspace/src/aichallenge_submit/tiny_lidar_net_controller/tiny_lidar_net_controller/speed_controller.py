"""Longitudinal controller for the TinyLiDARNet submission baseline."""


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
