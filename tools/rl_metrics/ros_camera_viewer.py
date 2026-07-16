#!/usr/bin/env python3
"""Small ROS 2 camera image viewer for AI Challenge monitoring captures."""

from __future__ import annotations

import argparse
import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class CameraViewer(Node):
    def __init__(self, topic: str, window_name: str, width: int) -> None:
        super().__init__("ai_challenge_camera_viewer")
        self._bridge = CvBridge()
        self._window_name = window_name
        self._width = width
        self._frame_count = 0
        self._started = time.monotonic()
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, width, int(width * 9 / 16))
        x = os.environ.get("CAMERA_VIEW_X")
        y = os.environ.get("CAMERA_VIEW_Y")
        if x is not None and y is not None:
            cv2.moveWindow(self._window_name, int(x), int(y))
        self.create_subscription(Image, topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f"subscribed: {topic}")

    def _on_image(self, msg: Image) -> None:
        self._frame_count += 1
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"cv_bridge failed: {exc}")
            return

        if self._width and image.shape[1] != self._width:
            scale = self._width / image.shape[1]
            image = cv2.resize(image, (self._width, int(image.shape[0] * scale)))

        elapsed = max(1e-6, time.monotonic() - self._started)
        fps = self._frame_count / elapsed
        label = f"{self._window_name}  frame={self._frame_count}  avg={fps:.1f} fps"
        cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(image, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(self._window_name, image)
        cv2.waitKey(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/sensing/camera/image_raw")
    parser.add_argument("--window-name", default="PilotNet Camera")
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    rclpy.init()
    node = CameraViewer(args.topic, args.window_name, args.width)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
