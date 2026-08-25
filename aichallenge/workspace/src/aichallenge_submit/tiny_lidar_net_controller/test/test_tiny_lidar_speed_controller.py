import math

import pytest

import numpy as np

from tiny_lidar_net_controller.speed_controller import (
    SpeedController,
    StuckDetector,
    TimeToCollisionGovernor,
    calculate_forward_clearance,
    detect_compact_forward_obstacle,
    select_target_speed,
    should_activate_predictive_avoidance,
)


def ttc_governor(**overrides):
    parameters = dict(
        activation_ttc_sec=3.0,
        minimum_ttc_sec=1.5,
        minimum_speed_scale=0.6,
        minimum_closing_speed_mps=0.5,
        rate_history_size=3,
        hold_steps=2,
    )
    parameters.update(overrides)
    return TimeToCollisionGovernor(**parameters)


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


def test_uses_stronger_braking_only_for_large_overspeed():
    adaptive = SpeedController(
        target_speed_mps=3.0,
        proportional_gain=0.8,
        min_acceleration=-0.2,
        max_acceleration=0.6,
        hard_braking_threshold_mps=0.3,
        hard_min_acceleration=-0.6,
    )
    assert adaptive.compute(3.2) == pytest.approx(-0.16)
    assert adaptive.compute(3.4) == pytest.approx(-0.32)
    assert adaptive.compute(4.0) == pytest.approx(-0.6)


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


def test_ttc_governor_does_not_react_to_a_single_range_jump():
    governor = ttc_governor()
    assert governor.compute(12.0, 0.0) == pytest.approx(1.0)
    assert governor.compute(8.0, 0.1) == pytest.approx(1.0)
    assert governor.compute(8.0, 0.2) == pytest.approx(1.0)


def test_predictive_avoidance_requires_a_straight_closing_path():
    common = {
        "predictive_scale": 0.5,
        "front_clearance_m": 5.0,
        "maximum_distance_m": 6.0,
        "maximum_steering": 0.08,
    }
    assert should_activate_predictive_avoidance(steering_angle=0.04, **common)
    assert not should_activate_predictive_avoidance(steering_angle=0.2, **common)
    assert not should_activate_predictive_avoidance(
        steering_angle=0.04,
        **{**common, "predictive_scale": 1.0},
    )
    assert not should_activate_predictive_avoidance(
        steering_angle=0.04,
        compact_obstacle_detected=False,
        **common,
    )


def test_compact_obstacle_detector_accepts_isolated_cluster_and_rejects_wall():
    angles = np.linspace(-math.pi / 4, math.pi / 4, 181)
    cluster = np.full(angles.shape, 20.0)
    cluster[np.abs(angles) <= math.radians(7.0)] = 5.0
    assert detect_compact_forward_obstacle(
        cluster,
        angles,
        range_min=0.05,
        range_max=25.0,
        maximum_distance_m=6.0,
    )

    wall = np.full(angles.shape, 20.0)
    wall[np.abs(angles) <= math.radians(35.0)] = 5.0
    assert not detect_compact_forward_obstacle(
        wall,
        angles,
        range_min=0.05,
        range_max=25.0,
        maximum_distance_m=6.0,
    )


def test_ttc_governor_slows_for_sustained_fast_approach():
    governor = ttc_governor()
    scales = [
        governor.compute(clearance, index * 0.1)
        for index, clearance in enumerate([10.0, 9.6, 9.2, 8.8])
    ]
    assert scales[-1] < 1.0
    assert governor.last_ttc_sec == pytest.approx(2.2)
    assert governor.intervention_ratio == pytest.approx(0.25)


def test_ttc_governor_can_command_a_full_yield_stop():
    governor = ttc_governor(
        activation_ttc_sec=2.5,
        minimum_ttc_sec=1.0,
        minimum_speed_scale=0.0,
    )
    scales = [
        governor.compute(clearance, index * 0.1)
        for index, clearance in enumerate([4.0, 3.5, 3.0, 2.5])
    ]
    assert governor.last_ttc_sec == pytest.approx(0.5)
    assert scales[-1] == pytest.approx(0.0)


def test_ttc_governor_ignores_slow_clearance_drift():
    governor = ttc_governor()
    scales = [
        governor.compute(clearance, index * 0.1)
        for index, clearance in enumerate([10.0, 9.98, 9.96, 9.94])
    ]
    assert scales[-1] == pytest.approx(1.0)
    assert math.isinf(governor.last_ttc_sec)


def test_ttc_governor_holds_a_recent_slowdown():
    governor = ttc_governor(rate_history_size=1, hold_steps=2)
    governor.compute(5.0, 0.0)
    slowed = governor.compute(4.0, 0.1)
    held = governor.compute(5.0, 0.2)
    assert slowed == pytest.approx(0.6)
    assert held == pytest.approx(slowed)


def test_ttc_governor_resets_after_a_stale_scan_gap():
    governor = ttc_governor(rate_history_size=1)
    governor.compute(10.0, 0.0)
    governor.compute(9.0, 0.1)
    assert governor.compute(1.0, 1.0) == pytest.approx(1.0)


def test_stuck_detector_does_not_trigger_during_initial_launch():
    detector = StuckDetector(trigger_duration_sec=1.0)
    assert not detector.compute(0.0, 0.0)
    assert not detector.compute(0.0, 2.0)


def test_stuck_detector_triggers_after_a_moving_vehicle_stops():
    detector = StuckDetector(trigger_duration_sec=1.0)
    assert not detector.compute(1.0, 0.0)
    assert not detector.compute(0.1, 1.0)
    assert detector.compute(0.1, 2.0)


def test_stuck_detector_clears_after_vehicle_moves_again():
    detector = StuckDetector(trigger_duration_sec=1.0)
    detector.compute(1.0, 0.0)
    detector.compute(0.1, 1.0)
    assert detector.compute(0.1, 2.0)
    assert not detector.compute(1.0, 2.1)
