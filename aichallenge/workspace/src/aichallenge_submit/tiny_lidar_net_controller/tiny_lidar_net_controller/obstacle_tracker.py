"""Temporal tracking for compact forward LiDAR obstacles."""

from collections import deque
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass(frozen=True)
class ObstacleCluster:
    angle_rad: float
    distance_m: float
    span_deg: float


@dataclass(frozen=True)
class TrackedThreat:
    angle_rad: float
    distance_m: float
    closing_speed_mps: float
    ttc_sec: float
    observations: int

    def speed_scale(
        self,
        *,
        activation_ttc_sec: float,
        minimum_ttc_sec: float,
        minimum_speed_scale: float,
    ) -> float:
        scale = (self.ttc_sec - minimum_ttc_sec) / (
            activation_ttc_sec - minimum_ttc_sec
        )
        return float(np.clip(scale, minimum_speed_scale, 1.0))


@dataclass
class _Track:
    identifier: int
    history: deque = field(default_factory=lambda: deque(maxlen=6))
    missed_updates: int = 0


def extract_compact_obstacles(
    ranges,
    angles,
    *,
    range_min: float,
    range_max: float,
    maximum_distance_m: float,
    half_angle_deg: float = 45.0,
    minimum_span_deg: float = 2.0,
    maximum_span_deg: float = 30.0,
    minimum_boundary_jump_m: float = 1.0,
    minimum_boundary_count: int = 2,
):
    """Extract isolated clusters while rejecting broad wall surfaces."""
    ranges = np.asarray(ranges, dtype=np.float32)
    angles = np.asarray(angles, dtype=np.float32)
    usable = np.where(np.isposinf(ranges), float(range_max), ranges)
    sector = (
        np.isfinite(usable)
        & (usable >= float(range_min))
        & (usable <= float(range_max))
        & (np.abs(angles) <= math.radians(half_angle_deg))
    )
    near_indices = np.flatnonzero(sector & (usable <= float(maximum_distance_m)))
    if near_indices.size == 0:
        return []

    runs = np.split(near_indices, np.flatnonzero(np.diff(near_indices) > 1) + 1)
    angle_step = float(np.median(np.abs(np.diff(angles)))) if angles.size > 1 else 0.0
    clusters = []
    for run in runs:
        start = int(run[0])
        end = int(run[-1])
        if start == 0 or end >= usable.size - 1:
            continue
        span_deg = math.degrees(abs(float(angles[end] - angles[start])) + angle_step)
        if not minimum_span_deg <= span_deg <= maximum_span_deg:
            continue
        if not (sector[start - 1] and sector[end + 1]):
            continue
        cluster_distance = float(np.median(usable[run]))
        left_jump = float(usable[start - 1]) - cluster_distance
        right_jump = float(usable[end + 1]) - cluster_distance
        boundary_count = int(left_jump >= minimum_boundary_jump_m) + int(
            right_jump >= minimum_boundary_jump_m
        )
        if boundary_count < minimum_boundary_count:
            continue
        clusters.append(
            ObstacleCluster(
                angle_rad=float(np.median(angles[run])),
                distance_m=cluster_distance,
                span_deg=span_deg,
            )
        )
    return clusters


class LidarObstacleTracker:
    """Associate compact clusters across scans and report confirmed threats."""

    def __init__(
        self,
        *,
        maximum_distance_m: float = 6.0,
        confirmation_hits: int = 3,
        maximum_missed_updates: int = 2,
        association_angle_deg: float = 10.0,
        association_distance_m: float = 1.5,
        minimum_closing_speed_mps: float = 0.3,
        maximum_track_age_sec: float = 0.5,
        minimum_boundary_count: int = 1,
    ) -> None:
        self.maximum_distance_m = float(maximum_distance_m)
        self.confirmation_hits = max(2, int(confirmation_hits))
        self.maximum_missed_updates = max(0, int(maximum_missed_updates))
        self.association_angle_rad = math.radians(association_angle_deg)
        self.association_distance_m = float(association_distance_m)
        self.minimum_closing_speed_mps = float(minimum_closing_speed_mps)
        self.maximum_track_age_sec = float(maximum_track_age_sec)
        self.minimum_boundary_count = int(np.clip(minimum_boundary_count, 1, 2))
        self._tracks = []
        self._next_identifier = 1
        self.last_threat = None
        self.confirmed_threat_count = 0

    def reset(self) -> None:
        self._tracks.clear()
        self.last_threat = None

    def update(
        self,
        ranges,
        angles,
        *,
        range_min: float,
        range_max: float,
        timestamp_sec: float,
    ):
        now = float(timestamp_sec)
        clusters = extract_compact_obstacles(
            ranges,
            angles,
            range_min=range_min,
            range_max=range_max,
            maximum_distance_m=self.maximum_distance_m,
            minimum_boundary_count=self.minimum_boundary_count,
        )
        unmatched_tracks = set(range(len(self._tracks)))
        unmatched_clusters = set(range(len(clusters)))

        candidates = []
        for track_index, track in enumerate(self._tracks):
            if not track.history:
                continue
            previous = track.history[-1][1]
            for cluster_index, cluster in enumerate(clusters):
                angle_error = abs(cluster.angle_rad - previous.angle_rad)
                distance_error = abs(cluster.distance_m - previous.distance_m)
                if (
                    angle_error <= self.association_angle_rad
                    and distance_error <= self.association_distance_m
                ):
                    score = (
                        angle_error / max(self.association_angle_rad, 1e-6)
                        + distance_error / max(self.association_distance_m, 1e-6)
                    )
                    candidates.append((score, track_index, cluster_index))

        for _, track_index, cluster_index in sorted(candidates):
            if track_index not in unmatched_tracks or cluster_index not in unmatched_clusters:
                continue
            track = self._tracks[track_index]
            track.history.append((now, clusters[cluster_index]))
            track.missed_updates = 0
            unmatched_tracks.remove(track_index)
            unmatched_clusters.remove(cluster_index)

        for track_index in unmatched_tracks:
            self._tracks[track_index].missed_updates += 1
        for cluster_index in unmatched_clusters:
            track = _Track(identifier=self._next_identifier)
            self._next_identifier += 1
            track.history.append((now, clusters[cluster_index]))
            self._tracks.append(track)

        self._tracks = [
            track
            for track in self._tracks
            if track.missed_updates <= self.maximum_missed_updates
            and track.history
            and now - track.history[-1][0] <= self.maximum_track_age_sec
        ]

        threats = []
        for track in self._tracks:
            if len(track.history) < self.confirmation_hits:
                continue
            first_time, first = track.history[0]
            last_time, last = track.history[-1]
            elapsed = last_time - first_time
            if elapsed <= 0.0:
                continue
            closing_speed = (first.distance_m - last.distance_m) / elapsed
            if closing_speed < self.minimum_closing_speed_mps:
                continue
            threats.append(
                TrackedThreat(
                    angle_rad=last.angle_rad,
                    distance_m=last.distance_m,
                    closing_speed_mps=closing_speed,
                    ttc_sec=last.distance_m / max(closing_speed, 1e-6),
                    observations=len(track.history),
                )
            )

        self.last_threat = min(threats, key=lambda threat: threat.ttc_sec) if threats else None
        if self.last_threat is not None:
            self.confirmed_threat_count += 1
        return self.last_threat
