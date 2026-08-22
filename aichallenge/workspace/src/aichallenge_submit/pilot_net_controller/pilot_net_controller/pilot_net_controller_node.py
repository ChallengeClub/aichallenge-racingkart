#!/usr/bin/env python3
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import VelocityReport

from pilot_net_controller_core import PilotNetCore
from pilot_net_controller.speed_controller import SpeedController
from pilot_net_controller.lidar_safety import LidarSafetyController


class PilotNetNode(Node):
    """ROS 2 Node for PilotNet autonomous driving control.

    Subscribes to camera Image messages, processes them using PilotNetCore,
    and publishes AckermannControlCommand messages.
    """

    def __init__(self):
        super().__init__('pilot_net_node')

        # --- Parameter Declaration (same pattern as TinyLidarNetNode) ---
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.image_height', 256)
        self.declare_parameter('model.image_width', 384)
        self.declare_parameter('model.output_dim', 2)
        self.declare_parameter('model.ckpt_path', '')
        self.declare_parameter('acceleration', 0.1)
        self.declare_parameter('control_mode', 'ai')
        self.declare_parameter('model.color_space', 'rgb')

        self.declare_parameter('model.crop_top_ratio', 0.0)
        self.declare_parameter('model.crop_bottom_ratio', 0.0)
        self.declare_parameter('speed_control.enabled', True)
        self.declare_parameter('speed_control.target_speed_mps', 2.0)
        self.declare_parameter('speed_control.proportional_gain', 0.8)
        self.declare_parameter('speed_control.min_acceleration', -0.2)
        self.declare_parameter('speed_control.max_acceleration', 0.6)
        self.declare_parameter('lidar_safety.enabled', True)
        self.declare_parameter('lidar_safety.activation_distance_m', 6.0)
        self.declare_parameter('lidar_safety.stop_distance_m', 0.5)
        self.declare_parameter('lidar_safety.max_steering_correction', 0.6)
        self.declare_parameter('lidar_safety.minimum_speed_scale', 0.25)
        self.declare_parameter('lidar_safety.minimum_steering_correction', 0.5)
        self.declare_parameter('lidar_safety.recovery_hold_steps', 12)
        self.declare_parameter('debug', False)

        # --- Initialization ---
        image_height = self.get_parameter('model.image_height').value
        image_width = self.get_parameter('model.image_width').value
        output_dim = self.get_parameter('model.output_dim').value
        ckpt_path = self.get_parameter('model.ckpt_path').value
        acceleration = self.get_parameter('acceleration').value
        control_mode = self.get_parameter('control_mode').value
        color_space = self.get_parameter('model.color_space').value
        crop_top_ratio = self.get_parameter('model.crop_top_ratio').value
        crop_bottom_ratio = self.get_parameter('model.crop_bottom_ratio').value
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
        self.speed_scale = 1.0
        self.steering_correction = 0.0

        self.debug = self.get_parameter('debug').value
        self.log_interval = self.get_parameter('log_interval_sec').value

        try:
            self.core = PilotNetCore(
                image_height=image_height,
                image_width=image_width,
                output_dim=output_dim,
                ckpt_path=ckpt_path,
                acceleration=acceleration,
                control_mode=control_mode,
                color_space=color_space,
                crop_top_ratio=crop_top_ratio,
                crop_bottom_ratio=crop_bottom_ratio,
            )
            self.get_logger().info(
                f"Core initialized. Image: {image_height}x{image_width}, "
                f"ColorSpace: {color_space}, "
                f"Crop: top={crop_top_ratio}/bottom={crop_bottom_ratio}, "
                f"OutputDim: {output_dim}, Mode: {control_mode}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize core logic: {e}")
            raise

        # --- Communication Setup ---
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_image = self.create_subscription(
            Image, "/image_raw", self.image_callback, qos
        )
        self.sub_velocity = self.create_subscription(
            VelocityReport,
            "/vehicle/status/velocity_status",
            self.velocity_callback,
            qos,
        )
        self.sub_scan = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            qos,
        )
        self.pub_control = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1
        )

        self.get_logger().info("PilotNetNode is ready.")

    def velocity_callback(self, msg: VelocityReport):
        """Store the latest wheel-odometry speed for longitudinal control."""
        self.current_speed_mps = float(msg.longitudinal_velocity)

    def scan_callback(self, msg: LaserScan):
        """Update the limited safety residual from the latest LiDAR scan."""
        if not self.lidar_safety_enabled:
            self.speed_scale = 1.0
            self.steering_correction = 0.0
            return
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        self.speed_scale, self.steering_correction, _ = self.lidar_safety.compute(
            msg.ranges, angles, msg.range_min, msg.range_max
        )

    def image_callback(self, msg: Image):
        """Callback for Image subscription."""
        start_time = time.monotonic()

        # 1. Convert ROS Image to NumPy array
        image = self._image_msg_to_numpy(msg)
        if image is None:
            return

        # 2. Process via Core Logic
        accel, steer = self.core.process(image)
        if self.speed_control_enabled:
            if self.current_speed_mps is None:
                accel = 0.0
            else:
                target_speed = self.speed_controller.target_speed_mps * self.speed_scale
                accel = self.speed_controller.compute(
                    self.current_speed_mps, target_speed_mps=target_speed
                )
        steer = float(np.clip(steer + self.steering_correction, -1.0, 1.0))

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

    def _image_msg_to_numpy(self, msg: Image) -> np.ndarray:
        """Converts a ROS Image message to a NumPy array (H, W, 3) RGB uint8."""
        try:
            # Get raw data as numpy array
            if msg.encoding == 'bgr8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
            elif msg.encoding == 'bgra8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif msg.encoding == 'rgba8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}", throttle_duration_sec=5.0)
                return None
            return img
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}", throttle_duration_sec=5.0)
            return None

    def _log_performance_metrics(self):
        """Logs performance metrics at fixed intervals (same as TinyLidarNetNode)."""
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
    node = PilotNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
