"""Small learned steering residual conditioned on LiDAR and wheel speed."""

from pathlib import Path

import numpy as np


def extract_residual_features(
    ranges: np.ndarray,
    nominal_steering: float,
    speed_mps: float,
    sector_count: int = 18,
    maximum_range_m: float = 30.0,
) -> np.ndarray:
    """Compress a scan into robust sector statistics plus vehicle state."""
    scan = np.asarray(ranges, dtype=np.float32).reshape(-1)
    scan = np.nan_to_num(
        scan, nan=0.0, posinf=maximum_range_m, neginf=0.0
    )
    scan = np.clip(scan, 0.0, maximum_range_m)
    sectors = np.array_split(scan, sector_count)
    near = np.asarray([np.percentile(values, 20) for values in sectors])
    mean = np.asarray([np.mean(values) for values in sectors])
    normalized_speed = abs(float(speed_mps)) / 3.5
    steer = float(nominal_steering)
    return np.concatenate(
        (
            near / maximum_range_m,
            mean / maximum_range_m,
            np.asarray(
                [steer, abs(steer), normalized_speed, steer * normalized_speed],
                dtype=np.float64,
            ),
        )
    )


class LearnedSpeedResidual:
    """Bounded ridge model whose correction fades to zero at normal speed."""

    def __init__(self, weights_path: str):
        weights = np.load(Path(weights_path))
        self.feature_mean = weights["feature_mean"]
        self.feature_scale = weights["feature_scale"]
        self.coefficients = weights["coefficients"]
        self.intercept = float(weights["intercept"])
        self.activation_speed_mps = float(weights["activation_speed_mps"])
        self.full_effect_speed_mps = float(weights["full_effect_speed_mps"])
        self.maximum_correction = float(weights["maximum_correction"])
        self.sector_count = int(weights["sector_count"])
        self.maximum_range_m = float(weights["maximum_range_m"])

    def speed_gate(self, speed_mps: float) -> float:
        span = max(1e-6, self.full_effect_speed_mps - self.activation_speed_mps)
        return float(
            np.clip((abs(speed_mps) - self.activation_speed_mps) / span, 0.0, 1.0)
        )

    def compute(
        self,
        ranges: np.ndarray,
        nominal_steering: float,
        speed_mps: float,
    ) -> float:
        gate = self.speed_gate(speed_mps)
        if gate <= 0.0:
            return float(nominal_steering)
        features = extract_residual_features(
            ranges,
            nominal_steering,
            speed_mps,
            sector_count=self.sector_count,
            maximum_range_m=self.maximum_range_m,
        )
        normalized = (features - self.feature_mean) / self.feature_scale
        raw_correction = float(normalized @ self.coefficients + self.intercept)
        correction = gate * float(
            np.clip(raw_correction, -self.maximum_correction, self.maximum_correction)
        )
        return float(nominal_steering + correction)
