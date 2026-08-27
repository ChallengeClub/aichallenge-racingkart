from pathlib import Path

import numpy as np

from tiny_lidar_net_controller.learned_speed_residual import (
    LearnedSpeedResidual,
    extract_residual_features,
)


def write_weights(path: Path, intercept: float = 0.2) -> None:
    feature_count = 18 * 2 + 4
    np.savez(
        path,
        feature_mean=np.zeros(feature_count),
        feature_scale=np.ones(feature_count),
        coefficients=np.zeros(feature_count),
        intercept=np.asarray(intercept),
        activation_speed_mps=np.asarray(2.3),
        full_effect_speed_mps=np.asarray(3.3),
        maximum_correction=np.asarray(0.12),
        sector_count=np.asarray(18),
        maximum_range_m=np.asarray(30.0),
    )


def test_features_are_finite_for_invalid_ranges():
    ranges = np.linspace(0.0, 35.0, 180)
    ranges[0] = np.nan
    ranges[1] = np.inf
    features = extract_residual_features(ranges, 0.1, 3.0)
    assert features.shape == (40,)
    assert np.all(np.isfinite(features))


def test_residual_is_neutral_at_normal_speed(tmp_path):
    weights = tmp_path / "weights.npz"
    write_weights(weights)
    model = LearnedSpeedResidual(str(weights))
    assert model.compute(np.ones(180), 0.1, 2.3) == 0.1


def test_residual_is_speed_gated_and_bounded(tmp_path):
    weights = tmp_path / "weights.npz"
    write_weights(weights)
    model = LearnedSpeedResidual(str(weights))
    # Raw 0.2 is clipped to 0.12; at 2.8 m/s the speed gate is 0.5.
    assert np.isclose(model.compute(np.ones(180), 0.1, 2.8), 0.16)
    assert np.isclose(model.compute(np.ones(180), 0.1, 3.5), 0.22)
