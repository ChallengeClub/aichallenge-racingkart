import numpy as np
import pytest

from pilot_net_controller.lidar_safety import LidarSafetyController


def make_controller():
    return LidarSafetyController(
        activation_distance_m=6.0,
        stop_distance_m=0.5,
        max_steering_correction=0.6,
        minimum_speed_scale=0.25,
    )


def scan(front=10.0, left=3.0, right=3.0):
    angles = np.linspace(-np.pi / 2, np.pi / 2, 181, dtype=np.float32)
    ranges = np.full_like(angles, front)
    ranges[(angles >= np.deg2rad(20)) & (angles <= np.deg2rad(80))] = left
    ranges[(angles <= np.deg2rad(-20)) & (angles >= np.deg2rad(-80))] = right
    return ranges, angles


def test_does_not_intervene_when_front_is_clear():
    ranges, angles = scan(front=10.0, left=1.0, right=5.0)
    scale, correction, clearance = make_controller().compute(ranges, angles, 0.05, 25.0)
    assert scale == pytest.approx(1.0)
    assert correction == pytest.approx(0.0)
    assert clearance == pytest.approx(10.0)


def test_turns_toward_open_left_side_and_slows_down():
    ranges, angles = scan(front=2.0, left=5.0, right=0.5)
    scale, correction, _ = make_controller().compute(ranges, angles, 0.05, 25.0)
    assert 0.0 < scale < 1.0
    assert correction > 0.0


def test_keeps_a_slow_recovery_crawl_at_close_obstacle():
    ranges, angles = scan(front=0.4, left=5.0, right=0.5)
    scale, _, _ = make_controller().compute(ranges, angles, 0.05, 25.0)
    assert scale == pytest.approx(0.25)


def test_holds_the_recovery_turn_after_front_becomes_clear():
    controller = LidarSafetyController(
        activation_distance_m=6.0,
        stop_distance_m=0.5,
        max_steering_correction=0.8,
        minimum_speed_scale=0.25,
        minimum_steering_correction=0.5,
        recovery_hold_steps=2,
    )
    blocked, angles = scan(front=2.0, left=0.5, right=5.0)
    _, first_correction, _ = controller.compute(blocked, angles, 0.05, 25.0)
    clear, angles = scan(front=10.0, left=10.0, right=10.0)
    _, held_correction, _ = controller.compute(clear, angles, 0.05, 25.0)
    assert first_correction <= -0.5
    assert held_correction == pytest.approx(first_correction)
