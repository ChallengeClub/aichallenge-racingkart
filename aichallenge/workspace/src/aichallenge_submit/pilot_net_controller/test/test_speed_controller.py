import pytest

from pilot_net_controller.speed_controller import SpeedController


def make_controller():
    return SpeedController(
        target_speed_mps=2.0,
        proportional_gain=0.8,
        min_acceleration=-0.2,
        max_acceleration=0.6,
    )


def test_accelerates_from_rest_with_upper_bound():
    assert make_controller().compute(0.0) == pytest.approx(0.6)


def test_reduces_acceleration_near_target_speed():
    assert make_controller().compute(1.5) == pytest.approx(0.4)
    assert make_controller().compute(2.0) == pytest.approx(0.0)


def test_applies_bounded_deceleration_when_overspeeding():
    assert make_controller().compute(3.0) == pytest.approx(-0.2)


def test_accepts_a_lower_dynamic_target():
    assert make_controller().compute(1.0, target_speed_mps=0.5) == pytest.approx(-0.2)


def test_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        SpeedController(
            target_speed_mps=2.0,
            proportional_gain=0.8,
            min_acceleration=0.5,
            max_acceleration=0.1,
        )
