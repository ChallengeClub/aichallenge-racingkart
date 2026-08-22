import pytest

import numpy as np

from tiny_lidar_net_controller.speed_controller import (
    SpeedController,
    calculate_forward_clearance,
    select_target_speed,
)


def controller():
    return SpeedController(
        target_speed_mps=2.5,
        proportional_gain=0.8,
        min_acceleration=-0.2,
        max_acceleration=0.6,
    )


def test_uses_launch_acceleration_when_stopped():
    assert controller().compute(0.0) == pytest.approx(0.6)


def test_reduces_acceleration_near_target():
    assert controller().compute(2.25) == pytest.approx(0.2)


def test_brakes_when_above_target():
    assert controller().compute(3.0) == pytest.approx(-0.2)


def test_accepts_a_safety_reduced_target():
    assert controller().compute(1.0, target_speed_mps=1.25) == pytest.approx(0.2)


def test_uses_higher_target_only_on_clear_straight():
    assert select_target_speed(
        base_speed_mps=2.5,
        straight_speed_mps=2.75,
        steering_angle=0.03,
        max_straight_steering=0.08,
        front_clearance_m=12.0,
        minimum_straight_clearance_m=12.0,
        safety_speed_scale=1.0,
    ) == pytest.approx(2.75)


def test_keeps_base_target_in_a_turn():
    assert select_target_speed(
        base_speed_mps=2.5,
        straight_speed_mps=2.75,
        steering_angle=0.2,
        max_straight_steering=0.08,
        front_clearance_m=10.0,
        minimum_straight_clearance_m=4.0,
        safety_speed_scale=1.0,
    ) == pytest.approx(2.5)


def test_safety_scale_overrides_straight_boost():
    assert select_target_speed(
        base_speed_mps=2.5,
        straight_speed_mps=2.75,
        steering_angle=0.0,
        max_straight_steering=0.08,
        front_clearance_m=2.0,
        minimum_straight_clearance_m=4.0,
        safety_speed_scale=0.5,
    ) == pytest.approx(1.25)


def test_forward_clearance_treats_infinite_return_as_open():
    angles = np.linspace(-0.2, 0.2, 5)
    ranges = np.array([2.0, np.inf, np.inf, np.inf, 2.0])
    assert calculate_forward_clearance(
        ranges,
        angles,
        range_min=0.05,
        range_max=25.0,
    ) == pytest.approx(25.0)


def test_forward_clearance_uses_low_quantile_for_obstacle():
    angles = np.linspace(-0.1, 0.1, 5)
    ranges = np.array([20.0, 20.0, 3.0, 20.0, 20.0])
    assert calculate_forward_clearance(
        ranges,
        angles,
        range_min=0.05,
        range_max=25.0,
    ) < 20.0
