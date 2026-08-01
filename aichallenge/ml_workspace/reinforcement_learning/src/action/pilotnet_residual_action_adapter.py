from __future__ import annotations

import sys
from typing import Any

import numpy as np
from gymnasium import spaces

from action.interfaces import ActionAdapter
from context.context_types import StepContext


class PilotNetResidualActionAdapter(ActionAdapter):
    """Camera-only PilotNet baseline with bounded SAC residual controls."""

    def __init__(
        self,
        *,
        package_path: str,
        checkpoint_path: str,
        steering_residual_scale: float = 0.0,
        acceleration_residual_scale: float = 0.0,
        max_speed_mps: float = 4.0,
        pilot_core: Any = None,
    ) -> None:
        if pilot_core is None:
            if package_path not in sys.path:
                sys.path.insert(0, package_path)
            from pilot_net_controller_core import PilotNetCore

            pilot_core = PilotNetCore(
                image_height=66,
                image_width=200,
                output_dim=2,
                ckpt_path=checkpoint_path,
                control_mode="ai",
                color_space="yuv",
                crop_top_ratio=0.375,
            )
        self._pilot_core = pilot_core
        self.steering_residual_scale = float(steering_residual_scale)
        self.acceleration_residual_scale = float(acceleration_residual_scale)
        self.max_speed_mps = float(max_speed_mps)

    @property
    def action_space(self) -> spaces.Box:
        return spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def adapt(self, action: np.ndarray, context: StepContext | None = None) -> dict[str, float]:
        image = None if context is None else context.env_state.get_value("camera_image")
        speed = 0.0 if context is None else max(
            0.0, float(context.env_state.get_value("vehicle_speed_mps", 0.0))
        )

        if image is None:
            baseline_acceleration, baseline_steering = 0.0, 0.0
        else:
            baseline_acceleration, baseline_steering = self._pilot_core.process(image)

        acceleration = baseline_acceleration + self.acceleration_residual_scale * float(action[1])
        if speed >= self.max_speed_mps:
            acceleration = min(acceleration, 0.0)
        steering = baseline_steering + self.steering_residual_scale * float(action[0])

        return {
            "steering": float(np.clip(steering, -1.0, 1.0)),
            "acceleration": float(np.clip(acceleration, 0.0, 1.0)),
        }
