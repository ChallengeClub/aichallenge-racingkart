import math

import numpy as np
import pytest

from tiny_lidar_net_controller.obstacle_tracker import LidarObstacleTracker


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
