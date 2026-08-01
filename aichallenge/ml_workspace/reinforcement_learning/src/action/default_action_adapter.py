import numpy as np
from gymnasium import spaces

from action.interfaces import ActionAdapter


class DefaultAWSIMActionAdapter(ActionAdapter):
    def __init__(
        self,
        min_steering: float = -1.0,
        max_steering: float = 1.0,
        min_accel: float = 0.0,
        max_accel: float = 1.0,
        sim_min_accel: float | None = None,
        sim_max_accel: float | None = None,
    ) -> None:
        self.min_steering = min_steering
        self.max_steering = max_steering
        self.min_accel = min_accel
        self.max_accel = max_accel
        self.sim_min_accel = min_accel if sim_min_accel is None else sim_min_accel
        self.sim_max_accel = max_accel if sim_max_accel is None else sim_max_accel

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(
            low=np.array([self.min_steering, self.min_accel], dtype=np.float32),
            high=np.array([self.max_steering, self.max_accel], dtype=np.float32),
            shape=(2,),
            dtype=np.float32,
        )

    def _to_sim_action(self, action: np.ndarray) -> dict[str, float]:
        steering = float(np.clip(action[0], self.min_steering, self.max_steering))
        policy_acceleration = float(np.clip(action[1], self.min_accel, self.max_accel))
        policy_span = self.max_accel - self.min_accel
        if policy_span <= 0.0:
            raise ValueError("max_accel must be greater than min_accel")
        ratio = (policy_acceleration - self.min_accel) / policy_span
        acceleration = self.sim_min_accel + ratio * (self.sim_max_accel - self.sim_min_accel)
        return {
            "steering": steering,
            "acceleration": float(acceleration),
        }

    def adapt(self, action: np.ndarray, context=None) -> dict[str, float]:
        return self._to_sim_action(action)
