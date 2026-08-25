#!/usr/bin/env python3
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport

from tiny_lidar_net_controller_core import TinyLidarNetCore
from tiny_lidar_net_controller.speed_controller import (
    SpeedController,
    StuckDetector,
    TimeToCollisionGovernor,
    calculate_forward_clearance,
    detect_compact_forward_obstacle,
    select_target_speed,
    should_activate_predictive_avoidance,
)
from tiny_lidar_net_controller.obstacle_tracker import (
    AdaptivePassingGate,
    LidarObstacleTracker,
)
from pilot_net_controller.lidar_safety import LidarSafetyController


class TinyLidarNetNode(Node):
    """ROS 2 Node for TinyLidarNet autonomous driving control.

    This node subscribes to LaserScan messages, processes them using the
    TinyLidarNetCore logic, and publishes AckermannControlCommand messages.
    """

    def __init__(self):
        super().__init__('tiny_lidar_net_node')

        # --- Parameter Declaration ---
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.output_dim', 2)
        self.declare_parameter('model.architecture', 'large')
        self.declare_parameter('model.ckpt_path', '')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('acceleration', 0.1)
        self.declare_parameter('control_mode', 'ai')
        self.declare_parameter('speed_control.enabled', True)
        self.declare_parameter('speed_control.target_speed_mps', 2.5)
        self.declare_parameter('speed_control.proportional_gain', 0.8)
        self.declare_parameter('speed_control.min_acceleration', -0.2)
        self.declare_parameter('speed_control.max_acceleration', 0.6)
        self.declare_parameter('speed_control.adaptive_braking_enabled', False)
        self.declare_parameter('speed_control.hard_braking_threshold_mps', 0.3)
        self.declare_parameter('speed_control.hard_min_acceleration', -0.6)
        self.declare_parameter('speed_control.straight_boost_enabled', True)
        self.declare_parameter('speed_control.straight_speed_mps', 2.75)
        self.declare_parameter('speed_control.max_straight_steering', 0.08)
        self.declare_parameter('speed_control.minimum_straight_clearance_m', 12.0)
        self.declare_parameter('speed_control.predictive_slowdown_enabled', True)
        self.declare_parameter('speed_control.activation_ttc_sec', 3.0)
        self.declare_parameter('speed_control.minimum_ttc_sec', 1.5)
        self.declare_parameter('speed_control.predictive_minimum_speed_scale', 0.6)
        self.declare_parameter('speed_control.predictive_avoidance_distance_m', 6.0)
        self.declare_parameter('speed_control.predictive_avoidance_max_steering', 0.08)
        self.declare_parameter('speed_control.minimum_closing_speed_mps', 0.5)
        self.declare_parameter('speed_control.ttc_rate_history_size', 3)
        self.declare_parameter('speed_control.ttc_hold_steps', 5)
        self.declare_parameter('npc_tracker.enabled', False)
        self.declare_parameter('npc_tracker.maximum_distance_m', 6.0)
        self.declare_parameter('npc_tracker.confirmation_hits', 3)
        self.declare_parameter('npc_tracker.maximum_missed_updates', 2)
        self.declare_parameter('npc_tracker.association_angle_deg', 10.0)
        self.declare_parameter('npc_tracker.association_distance_m', 1.5)
        self.declare_parameter('npc_tracker.minimum_closing_speed_mps', 0.3)
        self.declare_parameter('npc_tracker.maximum_track_age_sec', 0.5)
        self.declare_parameter('npc_tracker.minimum_boundary_count', 1)
        self.declare_parameter('npc_tracker.activation_ttc_sec', 3.0)
        self.declare_parameter('npc_tracker.minimum_ttc_sec', 1.2)
        self.declare_parameter('npc_tracker.minimum_speed_scale', 0.35)
        self.declare_parameter('npc_tracker.speed_scale_exponent', 1.0)
        self.declare_parameter('npc_tracker.adaptive_speed_scale_enabled', False)
        self.declare_parameter('npc_tracker.passing_speed_scale_exponent', 0.75)
        self.declare_parameter('npc_tracker.minimum_lateral_offset_m', 0.8)
        self.declare_parameter(
            'npc_tracker.minimum_lateral_separation_speed_mps', 0.2
        )
        self.declare_parameter('npc_tracker.adaptive_minimum_distance_m', 2.5)
        self.declare_parameter('npc_tracker.adaptive_confirmation_updates', 5)
        self.declare_parameter('npc_tracker.adaptive_maximum_duration_sec', 0.8)
        self.declare_parameter('npc_tracker.adaptive_cooldown_sec', 1.0)
        self.declare_parameter('npc_tracker.adaptive_minimum_ttc_sec', 1.8)
        self.declare_parameter('stuck_recovery.enabled', True)
        self.declare_parameter('stuck_recovery.stopped_speed_mps', 0.2)
        self.declare_parameter('stuck_recovery.moving_speed_mps', 0.8)
        self.declare_parameter('stuck_recovery.trigger_duration_sec', 2.0)
        self.declare_parameter('lidar_safety.enabled', True)
        self.declare_parameter('lidar_safety.activation_distance_m', 4.0)
        self.declare_parameter('lidar_safety.stop_distance_m', 0.5)
        self.declare_parameter('lidar_safety.max_steering_correction', 0.6)
        self.declare_parameter('lidar_safety.minimum_speed_scale', 0.5)
        self.declare_parameter('lidar_safety.minimum_steering_correction', 0.35)
        self.declare_parameter('lidar_safety.recovery_hold_steps', 6)
        self.declare_parameter('debug', False)

        # --- Initialization ---
        input_dim = self.get_parameter('model.input_dim').value
        output_dim = self.get_parameter('model.output_dim').value
        architecture = self.get_parameter('model.architecture').value
        ckpt_path = self.get_parameter('model.ckpt_path').value
        max_range = self.get_parameter('max_range').value
        acceleration = self.get_parameter('acceleration').value
        control_mode = self.get_parameter('control_mode').value
        self.speed_control_enabled = self.get_parameter('speed_control.enabled').value
        adaptive_braking_enabled = self.get_parameter(
            'speed_control.adaptive_braking_enabled'
        ).value
        self.speed_controller = SpeedController(
            target_speed_mps=self.get_parameter('speed_control.target_speed_mps').value,
            proportional_gain=self.get_parameter('speed_control.proportional_gain').value,
            min_acceleration=self.get_parameter('speed_control.min_acceleration').value,
            max_acceleration=self.get_parameter('speed_control.max_acceleration').value,
            hard_braking_threshold_mps=(
                self.get_parameter('speed_control.hard_braking_threshold_mps').value
                if adaptive_braking_enabled
                else None
            ),
            hard_min_acceleration=self.get_parameter(
                'speed_control.hard_min_acceleration'
            ).value,
        )
        self.current_speed_mps = None
        self.straight_boost_enabled = self.get_parameter(
            'speed_control.straight_boost_enabled'
        ).value
        self.straight_speed_mps = self.get_parameter(
            'speed_control.straight_speed_mps'
        ).value
        self.max_straight_steering = self.get_parameter(
            'speed_control.max_straight_steering'
        ).value
        self.minimum_straight_clearance_m = self.get_parameter(
            'speed_control.minimum_straight_clearance_m'
        ).value
        self.predictive_slowdown_enabled = self.get_parameter(
            'speed_control.predictive_slowdown_enabled'
        ).value
        self.ttc_governor = TimeToCollisionGovernor(
            activation_ttc_sec=self.get_parameter(
                'speed_control.activation_ttc_sec'
            ).value,
            minimum_ttc_sec=self.get_parameter('speed_control.minimum_ttc_sec').value,
            minimum_speed_scale=self.get_parameter(
                'speed_control.predictive_minimum_speed_scale'
            ).value,
            minimum_closing_speed_mps=self.get_parameter(
                'speed_control.minimum_closing_speed_mps'
            ).value,
            rate_history_size=self.get_parameter(
                'speed_control.ttc_rate_history_size'
            ).value,
            hold_steps=self.get_parameter('speed_control.ttc_hold_steps').value,
        )
        self.predictive_avoidance_distance_m = self.get_parameter(
            'speed_control.predictive_avoidance_distance_m'
        ).value
        self.predictive_avoidance_max_steering = self.get_parameter(
            'speed_control.predictive_avoidance_max_steering'
        ).value
        self.npc_tracker_enabled = self.get_parameter('npc_tracker.enabled').value
        self.npc_tracker = LidarObstacleTracker(
            maximum_distance_m=self.get_parameter(
                'npc_tracker.maximum_distance_m'
            ).value,
            confirmation_hits=self.get_parameter('npc_tracker.confirmation_hits').value,
            maximum_missed_updates=self.get_parameter(
                'npc_tracker.maximum_missed_updates'
            ).value,
            association_angle_deg=self.get_parameter(
                'npc_tracker.association_angle_deg'
            ).value,
            association_distance_m=self.get_parameter(
                'npc_tracker.association_distance_m'
            ).value,
            minimum_closing_speed_mps=self.get_parameter(
                'npc_tracker.minimum_closing_speed_mps'
            ).value,
            maximum_track_age_sec=self.get_parameter(
                'npc_tracker.maximum_track_age_sec'
            ).value,
            minimum_boundary_count=self.get_parameter(
                'npc_tracker.minimum_boundary_count'
            ).value,
        )
        self.npc_tracker_activation_ttc_sec = self.get_parameter(
            'npc_tracker.activation_ttc_sec'
        ).value
        self.npc_tracker_minimum_ttc_sec = self.get_parameter(
            'npc_tracker.minimum_ttc_sec'
        ).value
        self.npc_tracker_minimum_speed_scale = self.get_parameter(
            'npc_tracker.minimum_speed_scale'
        ).value
        self.npc_tracker_speed_scale_exponent = self.get_parameter(
            'npc_tracker.speed_scale_exponent'
        ).value
        self.npc_tracker_adaptive_speed_scale_enabled = self.get_parameter(
            'npc_tracker.adaptive_speed_scale_enabled'
        ).value
        self.npc_tracker_passing_speed_scale_exponent = self.get_parameter(
            'npc_tracker.passing_speed_scale_exponent'
        ).value
        self.npc_tracker_minimum_lateral_offset_m = self.get_parameter(
            'npc_tracker.minimum_lateral_offset_m'
        ).value
        self.npc_tracker_minimum_lateral_separation_speed_mps = self.get_parameter(
            'npc_tracker.minimum_lateral_separation_speed_mps'
        ).value
        self.npc_tracker_adaptive_minimum_distance_m = self.get_parameter(
            'npc_tracker.adaptive_minimum_distance_m'
        ).value
        self.npc_passing_gate = AdaptivePassingGate(
            confirmation_updates=self.get_parameter(
                'npc_tracker.adaptive_confirmation_updates'
            ).value,
            maximum_active_duration_sec=self.get_parameter(
                'npc_tracker.adaptive_maximum_duration_sec'
            ).value,
            cooldown_sec=self.get_parameter(
                'npc_tracker.adaptive_cooldown_sec'
            ).value,
            minimum_lateral_offset_m=self.npc_tracker_minimum_lateral_offset_m,
            minimum_lateral_separation_speed_mps=(
                self.npc_tracker_minimum_lateral_separation_speed_mps
            ),
            minimum_distance_m=self.npc_tracker_adaptive_minimum_distance_m,
            minimum_ttc_sec=self.get_parameter(
                'npc_tracker.adaptive_minimum_ttc_sec'
            ).value,
        )
        self.stuck_recovery_enabled = self.get_parameter(
            'stuck_recovery.enabled'
        ).value
        self.stuck_detector = StuckDetector(
            stopped_speed_mps=self.get_parameter(
                'stuck_recovery.stopped_speed_mps'
            ).value,
            moving_speed_mps=self.get_parameter(
                'stuck_recovery.moving_speed_mps'
            ).value,
            trigger_duration_sec=self.get_parameter(
                'stuck_recovery.trigger_duration_sec'
            ).value,
        )
        self.recovery_intervention_count = 0
        self.lidar_safety_enabled = self.get_parameter('lidar_safety.enabled').value
        self.lidar_safety = LidarSafetyController(
            activation_distance_m=self.get_parameter('lidar_safety.activation_distance_m').value,
            stop_distance_m=self.get_parameter('lidar_safety.stop_distance_m').value,
            max_steering_correction=self.get_parameter('lidar_safety.max_steering_correction').value,
            minimum_speed_scale=self.get_parameter('lidar_safety.minimum_speed_scale').value,
            minimum_steering_correction=self.get_parameter(
                'lidar_safety.minimum_steering_correction'
            ).value,
            recovery_hold_steps=self.get_parameter(
                'lidar_safety.recovery_hold_steps'
            ).value,
        )
        
        self.debug = self.get_parameter('debug').value
        self.log_interval = self.get_parameter('log_interval_sec').value

        try:
            self.core = TinyLidarNetCore(
                input_dim=input_dim,
                output_dim=output_dim,
                architecture=architecture,
                ckpt_path=ckpt_path,
                acceleration=acceleration,
                control_mode=control_mode,
                max_range=max_range
            )
            self.get_logger().info(
                f"Core initialized. Arch: {architecture}, MaxRange: {max_range}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize core logic: {e}")
            raise e

        # --- Communication Setup ---
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos
        )
        self.sub_velocity = self.create_subscription(
            VelocityReport,
            "/vehicle/status/velocity_status",
            self.velocity_callback,
            qos,
        )
        self.pub_control = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1
        )

        self.get_logger().info("TinyLidarNetNode is ready.")

    def velocity_callback(self, msg: VelocityReport):
        """Store permitted wheel-odometry speed for longitudinal control."""
        self.current_speed_mps = float(msg.longitudinal_velocity)

    def scan_callback(self, msg: LaserScan):
        """Callback for LaserScan subscription.

        Processes the scan data via the core logic and publishes a control command.

        Args:
            msg (LaserScan): The incoming ROS 2 LaserScan message.
        """
        start_time = time.monotonic()

        # 1. Convert ROS message to Numpy
        # We pass the raw array; the core logic handles NaN/Inf and normalization.
        ranges = np.array(msg.ranges, dtype=np.float32)
        now_monotonic = time.monotonic()
        speed_scale = 1.0
        steering_correction = 0.0
        force_recovery = (
            self.stuck_recovery_enabled
            and self.stuck_detector.compute(self.current_speed_mps, now_monotonic)
        )
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        front_clearance = calculate_forward_clearance(
            ranges,
            angles,
            range_min=msg.range_min,
            range_max=msg.range_max,
        )
        # Infer the nominal steering before applying safety residuals.  Early
        # avoidance is only appropriate on an otherwise straight path; on a
        # curve, rapidly changing wall ranges are not evidence of an NPC.
        accel, steer = self.core.process(ranges)
        predictive_scale = 1.0
        compact_obstacle_detected = False
        if self.predictive_slowdown_enabled:
            predictive_scale = self.ttc_governor.compute(
                front_clearance,
                now_monotonic,
            )
            speed_scale = min(speed_scale, predictive_scale)
            compact_obstacle_detected = detect_compact_forward_obstacle(
                ranges,
                angles,
                range_min=msg.range_min,
                range_max=msg.range_max,
                maximum_distance_m=self.predictive_avoidance_distance_m,
            )
        early_avoidance = should_activate_predictive_avoidance(
            predictive_scale=predictive_scale,
            front_clearance_m=front_clearance,
            maximum_distance_m=self.predictive_avoidance_distance_m,
            steering_angle=steer,
            maximum_steering=self.predictive_avoidance_max_steering,
            compact_obstacle_detected=compact_obstacle_detected,
        )
        tracked_threat = None
        tracked_scale = 1.0
        if self.npc_tracker_enabled:
            tracked_threat = self.npc_tracker.update(
                ranges,
                angles,
                range_min=msg.range_min,
                range_max=msg.range_max,
                timestamp_sec=now_monotonic,
            )
            passing_active = False
            if self.npc_tracker_adaptive_speed_scale_enabled:
                passing_active = self.npc_passing_gate.update(
                    tracked_threat,
                    now_monotonic,
                )
            if tracked_threat is not None:
                scale_exponent = self.npc_tracker_speed_scale_exponent
                if passing_active:
                    scale_exponent = self.npc_tracker_passing_speed_scale_exponent
                tracked_scale = tracked_threat.speed_scale(
                    activation_ttc_sec=self.npc_tracker_activation_ttc_sec,
                    minimum_ttc_sec=self.npc_tracker_minimum_ttc_sec,
                    minimum_speed_scale=self.npc_tracker_minimum_speed_scale,
                    exponent=scale_exponent,
                )
                speed_scale = min(speed_scale, tracked_scale)
                early_avoidance = True
        if self.lidar_safety_enabled:
            speed_scale, steering_correction, _ = self.lidar_safety.compute(
                ranges,
                angles,
                msg.range_min,
                msg.range_max,
                force_recovery=force_recovery,
                early_activation=early_avoidance,
            )
            speed_scale = min(speed_scale, predictive_scale, tracked_scale)
            if force_recovery:
                self.recovery_intervention_count += 1

        # 2. Apply longitudinal control and the lateral safety residual.
        if self.speed_control_enabled:
            accel = (
                0.0
                if self.current_speed_mps is None
                else self.speed_controller.compute(
                    self.current_speed_mps,
                    target_speed_mps=select_target_speed(
                        base_speed_mps=self.speed_controller.target_speed_mps,
                        straight_speed_mps=(
                            self.straight_speed_mps
                            if self.straight_boost_enabled
                            else self.speed_controller.target_speed_mps
                        ),
                        steering_angle=steer,
                        max_straight_steering=self.max_straight_steering,
                        front_clearance_m=front_clearance,
                        minimum_straight_clearance_m=(
                            self.minimum_straight_clearance_m
                        ),
                        safety_speed_scale=speed_scale,
                    ),
                )
            )
        steer = float(np.clip(steer + steering_correction, -1.0, 1.0))

        # 3. Publish Command
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = float(accel)
        cmd.lateral.steering_tire_angle = float(steer)
        self.pub_control.publish(cmd)

        # 4. Debug Logging
        if self.debug:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            self.inference_times.append(duration_ms)
            self._log_performance_metrics()

    def _log_performance_metrics(self):
        """Logs internal performance metrics at fixed intervals."""
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_log_time).nanoseconds / 1e9

        if elapsed_sec > self.log_interval:
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                max_time = np.max(self.inference_times)
                fps = 1000.0 / avg_time if avg_time > 0 else 0.0

                self.get_logger().info(
                    f"DEBUG: Avg Inference: {avg_time:.2f}ms ({fps:.2f}Hz) | "
                    f"Max: {max_time:.2f}ms | "
                    f"TTC: {self.ttc_governor.last_ttc_sec:.2f}s | "
                    f"TTC intervention: {self.ttc_governor.intervention_ratio:.1%} | "
                    f"Recovery scans: {self.recovery_intervention_count} | "
                    f"Tracked threats: {self.npc_tracker.confirmed_threat_count}"
                    f" | Passing activations: {self.npc_passing_gate.activation_count}"
                )
                self.inference_times.clear()
            
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = TinyLidarNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
