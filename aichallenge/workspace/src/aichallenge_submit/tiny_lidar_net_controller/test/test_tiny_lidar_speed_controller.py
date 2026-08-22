import pytest

from tiny_lidar_net_controller.speed_controller import SpeedController


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
