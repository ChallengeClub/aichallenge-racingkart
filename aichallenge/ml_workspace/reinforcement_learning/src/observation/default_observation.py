from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces
import cv2

from context.context_types import StepContext
from observation.interfaces import ObservationBuilder

# ORIGINAL SIZE
# IMAGE_HEIGHT = 256
# IMAGE_WIDTH = 384

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

IMAGE_CHANNELS = 3


class ImageSpeedObservationBuilder(ObservationBuilder):
    @property
    def observation_space(self) -> spaces.Dict:
        return spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
                    dtype=np.uint8,
                ),
                "speed": spaces.Box(
                    low=0.0,
                    high=np.finfo(np.float32).max,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

    def build(self, context: StepContext) -> tuple[dict[str, np.ndarray], StepContext]:
        image = context.env_state.get_value("camera_image")
        if image is None:
            image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8)
        else:
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

        speed_value = float(context.env_state.get_value("vehicle_speed_mps", 0.0))
        speed = np.array([max(0.0, speed_value)], dtype=np.float32)

        observation = {
            "image": image,
            "speed": speed,
        }
        return observation, context


class RacelineImageSpeedObservationBuilder(ImageSpeedObservationBuilder):
    """Image/speed observation augmented with ego-relative raceline points."""

    def __init__(
        self,
        raceline_csv_path: str,
        lookahead_indices: list[int] | None = None,
        position_scale_m: float = 30.0,
        loop_start_index: int = 0,
    ) -> None:
        path = Path(raceline_csv_path)
        if not path.is_file():
            raise ValueError(f"Raceline CSV not found: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 2:
            raise ValueError(f"Raceline CSV needs at least two points: {path}")

        self._raceline = np.asarray(
            [[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float64
        )
        self._lookahead_indices = tuple(lookahead_indices or [0, 3, 8, 15])
        self._position_scale_m = float(position_scale_m)
        self._previous_index: int | None = None
        self._loop_start_index = int(loop_start_index)

    def _wrap_index(self, index: int) -> int:
        if self._loop_start_index <= index < len(self._raceline):
            return index
        loop_length = len(self._raceline) - self._loop_start_index
        return self._loop_start_index + (index - self._loop_start_index) % loop_length

    @property
    def observation_space(self) -> spaces.Dict:
        base = super().observation_space.spaces
        return spaces.Dict(
            {
                **base,
                "raceline": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(2 * len(self._lookahead_indices),),
                    dtype=np.float32,
                ),
            }
        )

    def reset(self) -> None:
        self._previous_index = None

    @staticmethod
    def _yaw_from_context(context: StepContext) -> float:
        x = float(context.env_state.get_value("kinematic_orientation_x", 0.0))
        y = float(context.env_state.get_value("kinematic_orientation_y", 0.0))
        z = float(context.env_state.get_value("kinematic_orientation_z", 0.0))
        w = float(context.env_state.get_value("kinematic_orientation_w", 1.0))
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def build(self, context: StepContext) -> tuple[dict[str, np.ndarray], StepContext]:
        observation, context = super().build(context)
        raw_x = context.env_state.get_value("kinematic_pose_x_m")
        raw_y = context.env_state.get_value("kinematic_pose_y_m")
        if raw_x is None or raw_y is None:
            observation["raceline"] = np.zeros(
                2 * len(self._lookahead_indices), dtype=np.float32
            )
            context.info["raceline_valid"] = False
            context.info["raceline_progress_delta"] = 0
            context.info["cross_track_error_m"] = 0.0
            return observation, context

        x = float(raw_x)
        y = float(raw_y)
        position = np.asarray([x, y], dtype=np.float64)
        if self._previous_index is None:
            squared_distances = np.sum((self._raceline - position) ** 2, axis=1)
            nearest_index = int(np.argmin(squared_distances))
        else:
            candidates = [
                self._wrap_index(self._previous_index + offset) for offset in range(-3, 11)
            ]
            nearest_index = min(
                candidates,
                key=lambda index: float(np.sum((self._raceline[index] - position) ** 2)),
            )
        cross_track_error = float(np.linalg.norm(self._raceline[nearest_index] - position))

        yaw = self._yaw_from_context(context)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        relative_points: list[float] = []
        for offset in self._lookahead_indices:
            point = self._raceline[self._wrap_index(nearest_index + offset)] - position
            forward = cos_yaw * point[0] + sin_yaw * point[1]
            left = -sin_yaw * point[0] + cos_yaw * point[1]
            relative_points.extend(
                [
                    float(np.clip(forward / self._position_scale_m, -1.0, 1.0)),
                    float(np.clip(left / self._position_scale_m, -1.0, 1.0)),
                ]
            )

        progress_delta = 0
        if self._previous_index is not None:
            progress_delta = nearest_index - self._previous_index
            half = (len(self._raceline) - self._loop_start_index) // 2
            if progress_delta < -half:
                progress_delta += len(self._raceline) - self._loop_start_index
            elif progress_delta > half:
                progress_delta -= len(self._raceline) - self._loop_start_index
        self._previous_index = nearest_index

        observation["raceline"] = np.asarray(relative_points, dtype=np.float32)
        context.info["raceline_index"] = nearest_index
        context.info["raceline_valid"] = True
        context.info["raceline_progress_delta"] = progress_delta
        context.info["cross_track_error_m"] = cross_track_error
        return observation, context
