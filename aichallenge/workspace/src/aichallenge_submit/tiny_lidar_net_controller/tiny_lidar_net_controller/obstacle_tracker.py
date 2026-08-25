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
    lateral_separation_speed_mps: float = 0.0

    @property
    def lateral_offset_m(self) -> float:
        """Absolute lateral offset of the obstacle center in vehicle coordinates."""
        return abs(self.distance_m * math.sin(self.angle_rad))

    def adaptive_speed_scale_exponent(
        self,
        *,
        nominal_exponent: float,
        passing_exponent: float,
        minimum_lateral_offset_m: float,
        minimum_lateral_separation_speed_mps: float,
        minimum_distance_m: float,
    ) -> float:
        """Relax slowdown only after a tracked obstacle is moving laterally away."""
        safely_passing = (
            self.distance_m >= float(minimum_distance_m)
            and self.lateral_offset_m >= float(minimum_lateral_offset_m)
            and self.lateral_separation_speed_mps
            >= float(minimum_lateral_separation_speed_mps)
        )
        return float(passing_exponent if safely_passing else nominal_exponent)

    def is_safely_passing(
        self,
        *,
        minimum_lateral_offset_m: float,
        minimum_lateral_separation_speed_mps: float,
        minimum_distance_m: float,
        minimum_ttc_sec: float,
    ) -> bool:
        """Return whether geometry supports a conservative passing state."""
        return (
            self.distance_m >= float(minimum_distance_m)
            and self.ttc_sec >= float(minimum_ttc_sec)
            and self.lateral_offset_m >= float(minimum_lateral_offset_m)
            and self.lateral_separation_speed_mps
            >= float(minimum_lateral_separation_speed_mps)
        )

    def speed_scale(
        self,
        *,
        activation_ttc_sec: float,
        minimum_ttc_sec: float,
        minimum_speed_scale: float,
        exponent: float = 1.0,
    ) -> float:
        if exponent <= 0.0:
            raise ValueError("exponent must be positive")
        linear_scale = (self.ttc_sec - minimum_ttc_sec) / (
            activation_ttc_sec - minimum_ttc_sec
        )
        scale = np.clip(linear_scale, 0.0, 1.0) ** float(exponent)
        return float(np.clip(scale, minimum_speed_scale, 1.0))


@dataclass
class _Track:
    identifier: int
    history: deque = field(default_factory=lambda: deque(maxlen=6))
    missed_updates: int = 0


class AdaptivePassingGate:
    """Confirm safe passing geometry and bound any relaxed-speed interval."""

    def __init__(
        self,
        *,
        confirmation_updates: int = 5,
        maximum_active_duration_sec: float = 0.8,
        cooldown_sec: float = 1.0,
        minimum_lateral_offset_m: float = 0.8,
        minimum_lateral_separation_speed_mps: float = 0.2,
        minimum_distance_m: float = 2.5,
        minimum_ttc_sec: float = 1.8,
    ) -> None:
        self.confirmation_updates = max(1, int(confirmation_updates))
        self.maximum_active_duration_sec = max(
            0.0, float(maximum_active_duration_sec)
        )
        self.cooldown_sec = max(0.0, float(cooldown_sec))
        self.minimum_lateral_offset_m = float(minimum_lateral_offset_m)
        self.minimum_lateral_separation_speed_mps = float(
            minimum_lateral_separation_speed_mps
        )
        self.minimum_distance_m = float(minimum_distance_m)
        self.minimum_ttc_sec = float(minimum_ttc_sec)
        self._confirmation_count = 0
        self._active_since = None
        self._cooldown_until = None
        self.activation_count = 0
        self.active_update_count = 0

    @property
    def active(self) -> bool:
        return self._active_since is not None

    def reset(self) -> None:
        self._confirmation_count = 0
        self._active_since = None
        self._cooldown_until = None

    def _eligible(self, threat: TrackedThreat | None) -> bool:
        return threat is not None and threat.is_safely_passing(
            minimum_lateral_offset_m=self.minimum_lateral_offset_m,
            minimum_lateral_separation_speed_mps=(
                self.minimum_lateral_separation_speed_mps
            ),
            minimum_distance_m=self.minimum_distance_m,
            minimum_ttc_sec=self.minimum_ttc_sec,
        )

    def _deactivate(self, now: float) -> None:
        self._active_since = None
        self._confirmation_count = 0
        self._cooldown_until = now + self.cooldown_sec

    def update(self, threat: TrackedThreat | None, timestamp_sec: float) -> bool:
        now = float(timestamp_sec)
        eligible = self._eligible(threat)

        if self.active:
            expired = (
                now - self._active_since >= self.maximum_active_duration_sec
            )
            if not eligible or expired:
                self._deactivate(now)
                return False
            self.active_update_count += 1
            return True

        if self._cooldown_until is not None and now < self._cooldown_until:
            self._confirmation_count = 0
            return False
        if not eligible:
            self._confirmation_count = 0
            return False

        self._confirmation_count += 1
        if self._confirmation_count < self.confirmation_updates:
            return False

        self._confirmation_count = 0
        self._active_since = now
        self.activation_count += 1
        self.active_update_count += 1
        return True


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
            first_lateral_offset = abs(
                first.distance_m * math.sin(first.angle_rad)
            )
            last_lateral_offset = abs(last.distance_m * math.sin(last.angle_rad))
            threats.append(
                TrackedThreat(
                    angle_rad=last.angle_rad,
                    distance_m=last.distance_m,
                    closing_speed_mps=closing_speed,
                    ttc_sec=last.distance_m / max(closing_speed, 1e-6),
                    observations=len(track.history),
                    lateral_separation_speed_mps=(
                        last_lateral_offset - first_lateral_offset
                    )
                    / elapsed,
                )
            )

        self.last_threat = min(threats, key=lambda threat: threat.ttc_sec) if threats else None
        if self.last_threat is not None:
            self.confirmed_threat_count += 1
        return self.last_threat
