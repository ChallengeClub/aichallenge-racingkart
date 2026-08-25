import math

import numpy as np
import pytest

from tiny_lidar_net_controller.obstacle_tracker import (
    AdaptivePassingGate,
    LidarObstacleTracker,
    TrackedThreat,
)


ANGLES = np.linspace(-math.pi / 4, math.pi / 4, 181)


def scan_with_cluster(distance, center_deg=0.0, half_width_deg=7.0):
    ranges = np.full(ANGLES.shape, 20.0)
    center = math.radians(center_deg)
    half_width = math.radians(half_width_deg)
    ranges[np.abs(ANGLES - center) <= half_width] = distance
    return ranges


def update(tracker, ranges, timestamp):
    return tracker.update(
        ranges,
        ANGLES,
        range_min=0.05,
        range_max=25.0,
        timestamp_sec=timestamp,
    )


def test_requires_multiple_consistent_observations():
    tracker = LidarObstacleTracker(confirmation_hits=3)
    assert update(tracker, scan_with_cluster(5.0), 0.0) is None
    assert update(tracker, scan_with_cluster(4.7), 0.1) is None
    threat = update(tracker, scan_with_cluster(4.4), 0.2)
    assert threat is not None
    assert threat.closing_speed_mps == pytest.approx(3.0)
    assert threat.ttc_sec == pytest.approx(4.4 / 3.0)


def test_stationary_compact_object_is_not_a_closing_threat():
    tracker = LidarObstacleTracker(confirmation_hits=3)
    for index in range(4):
        threat = update(tracker, scan_with_cluster(5.0), index * 0.1)
    assert threat is None


def test_broad_wall_is_not_tracked_as_a_compact_obstacle():
    tracker = LidarObstacleTracker(confirmation_hits=3)
    wall = scan_with_cluster(5.0, half_width_deg=35.0)
    for index in range(4):
        threat = update(tracker, wall, index * 0.1)
    assert threat is None


def test_one_sided_cluster_can_be_confirmed_when_it_closes():
    tracker = LidarObstacleTracker(confirmation_hits=3, minimum_boundary_count=1)
    for index, distance in enumerate([5.7, 5.5, 5.3]):
        ranges = scan_with_cluster(distance)
        # The left edge blends into a surface just beyond the tracking range,
        # while the right edge remains clearly isolated.
        ranges[ANGLES < math.radians(-7.0)] = 6.1
        threat = update(tracker, ranges, index * 0.1)
    assert threat is not None


def test_angularly_jumping_clusters_do_not_form_one_track():
    tracker = LidarObstacleTracker(confirmation_hits=3, association_angle_deg=8.0)
    assert update(tracker, scan_with_cluster(5.5, center_deg=-20.0), 0.0) is None
    assert update(tracker, scan_with_cluster(5.0, center_deg=0.0), 0.1) is None
    assert update(tracker, scan_with_cluster(4.5, center_deg=20.0), 0.2) is None


def test_threat_speed_scale_reaches_configured_floor():
    tracker = LidarObstacleTracker(confirmation_hits=3)
    update(tracker, scan_with_cluster(3.0), 0.0)
    update(tracker, scan_with_cluster(2.5), 0.1)
    threat = update(tracker, scan_with_cluster(2.0), 0.2)
    assert threat.speed_scale(
        activation_ttc_sec=3.0,
        minimum_ttc_sec=1.2,
        minimum_speed_scale=0.35,
    ) == pytest.approx(0.35)


def test_speed_scale_exponent_preserves_more_speed_at_moderate_ttc():
    threat = TrackedThreat(
        angle_rad=0.0,
        distance_m=4.2,
        closing_speed_mps=2.0,
        ttc_sec=2.1,
        observations=4,
    )
    linear = threat.speed_scale(
        activation_ttc_sec=3.0,
        minimum_ttc_sec=1.2,
        minimum_speed_scale=0.35,
    )
    progressive = threat.speed_scale(
        activation_ttc_sec=3.0,
        minimum_ttc_sec=1.2,
        minimum_speed_scale=0.35,
        exponent=0.5,
    )
    assert linear == pytest.approx(0.5)
    assert progressive == pytest.approx(math.sqrt(0.5))


def test_speed_scale_exponent_keeps_critical_ttc_floor():
    threat = TrackedThreat(0.0, 1.0, 2.0, 0.5, 4)
    assert threat.speed_scale(
        activation_ttc_sec=3.0,
        minimum_ttc_sec=1.2,
        minimum_speed_scale=0.35,
        exponent=0.5,
    ) == pytest.approx(0.35)


def test_adaptive_scale_relaxes_only_for_a_laterally_diverging_threat():
    passing = TrackedThreat(
        angle_rad=math.radians(20.0),
        distance_m=4.0,
        closing_speed_mps=1.5,
        ttc_sec=2.0,
        observations=4,
        lateral_separation_speed_mps=0.4,
    )
    assert passing.lateral_offset_m > 0.8
    assert passing.adaptive_speed_scale_exponent(
        nominal_exponent=1.0,
        passing_exponent=0.75,
        minimum_lateral_offset_m=0.8,
        minimum_lateral_separation_speed_mps=0.2,
        minimum_distance_m=2.5,
    ) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "distance,angle_deg,separation_speed",
    [
        (4.0, 5.0, 0.4),  # Still near the predicted path.
        (4.0, 20.0, -0.1),  # Moving back toward the path.
        (2.0, 30.0, 0.4),  # Too close to relax longitudinal safety.
    ],
)
def test_adaptive_scale_keeps_linear_response_for_unsafe_geometry(
    distance, angle_deg, separation_speed
):
    threat = TrackedThreat(
        math.radians(angle_deg),
        distance,
        1.5,
        2.0,
        4,
        separation_speed,
    )
    assert threat.adaptive_speed_scale_exponent(
        nominal_exponent=1.0,
        passing_exponent=0.75,
        minimum_lateral_offset_m=0.8,
        minimum_lateral_separation_speed_mps=0.2,
        minimum_distance_m=2.5,
    ) == pytest.approx(1.0)


def safely_passing_threat(ttc_sec=2.2):
    return TrackedThreat(
        angle_rad=math.radians(20.0),
        distance_m=4.0,
        closing_speed_mps=1.5,
        ttc_sec=ttc_sec,
        observations=6,
        lateral_separation_speed_mps=0.4,
    )


def test_passing_gate_requires_consecutive_safe_updates():
    gate = AdaptivePassingGate(confirmation_updates=3)
    threat = safely_passing_threat()
    assert not gate.update(threat, 0.0)
    assert not gate.update(threat, 0.1)
    assert gate.update(threat, 0.2)
    assert gate.activation_count == 1


def test_passing_gate_immediately_exits_when_ttc_becomes_critical():
    gate = AdaptivePassingGate(confirmation_updates=1, minimum_ttc_sec=1.8)
    assert gate.update(safely_passing_threat(), 0.0)
    assert not gate.update(safely_passing_threat(ttc_sec=1.7), 0.1)


def test_passing_gate_limits_duration_and_applies_cooldown():
    gate = AdaptivePassingGate(
        confirmation_updates=1,
        maximum_active_duration_sec=0.5,
        cooldown_sec=1.0,
    )
    threat = safely_passing_threat()
    assert gate.update(threat, 0.0)
    assert gate.update(threat, 0.4)
    assert not gate.update(threat, 0.5)
    assert not gate.update(threat, 1.4)
    assert gate.update(threat, 1.5)


def test_passing_gate_resets_confirmation_after_unsafe_update():
    gate = AdaptivePassingGate(confirmation_updates=2)
    safe = safely_passing_threat()
    unsafe = TrackedThreat(0.0, 4.0, 1.5, 2.2, 6, 0.0)
    assert not gate.update(safe, 0.0)
    assert not gate.update(unsafe, 0.1)
    assert not gate.update(safe, 0.2)
    assert gate.update(safe, 0.3)
