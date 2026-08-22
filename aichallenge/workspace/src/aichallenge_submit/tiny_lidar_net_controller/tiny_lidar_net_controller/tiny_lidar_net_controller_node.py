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
from tiny_lidar_net_controller.speed_controller import SpeedController
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
        self.speed_controller = SpeedController(
            target_speed_mps=self.get_parameter('speed_control.target_speed_mps').value,
            proportional_gain=self.get_parameter('speed_control.proportional_gain').value,
            min_acceleration=self.get_parameter('speed_control.min_acceleration').value,
            max_acceleration=self.get_parameter('speed_control.max_acceleration').value,
        )
        self.current_speed_mps = None
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
        speed_scale = 1.0
        steering_correction = 0.0
        if self.lidar_safety_enabled:
            angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
            speed_scale, steering_correction, _ = self.lidar_safety.compute(
                ranges, angles, msg.range_min, msg.range_max
            )

        # 2. Process via Core Logic
        accel, steer = self.core.process(ranges)
        if self.speed_control_enabled:
            accel = (
                0.0
                if self.current_speed_mps is None
                else self.speed_controller.compute(
                    self.current_speed_mps,
                    target_speed_mps=(
                        self.speed_controller.target_speed_mps * speed_scale
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
                    f"Max: {max_time:.2f}ms"
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
