#!/usr/bin/env python3
"""Semi-realtime VLM commentary node for AI Challenge camera and vehicle topics."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request
import wave

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import AccelWithCovarianceStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

try:
    from autoware_auto_control_msgs.msg import AckermannControlCommand
except Exception:  # pragma: no cover - only for environments without Autoware messages
    AckermannControlCommand = None


def encode_jpeg_b64(image) -> str:
    ok, data = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        raise RuntimeError("failed to encode image")
    return base64.b64encode(data.tobytes()).decode("ascii")


def preprocess_image(image, mode: str):
    if mode == "full":
        return image
    if mode == "full_256x144":
        return cv2.resize(image, (256, 144), interpolation=cv2.INTER_AREA)
    if mode == "full_320x180":
        return cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
    upper = image[: image.shape[0] // 2, :]
    if mode == "upper_half":
        return upper
    if mode == "upper_320x160":
        return cv2.resize(upper, (320, 160), interpolation=cv2.INTER_AREA)
    if mode == "upper_160x80":
        return cv2.resize(upper, (160, 80), interpolation=cv2.INTER_AREA)
    lower = image[image.shape[0] // 2 :, :]
    if mode == "lower_half":
        return lower
    if mode == "lower_160x80":
        return cv2.resize(lower, (160, 80), interpolation=cv2.INTER_AREA)
    if mode == "lower_80x40":
        return cv2.resize(lower, (80, 40), interpolation=cv2.INTER_AREA)
    raise ValueError(f"unknown preprocess mode: {mode}")


def ollama_chat(base_url: str, payload: dict, timeout: int = 180) -> str:
    payload = {
        **payload,
        "stream": False,
        "options": {
            "temperature": payload.get("options", {}).get("temperature", 0.2),
            "num_predict": payload.get("options", {}).get("num_predict", 80),
        },
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("message", {}).get("content", "").strip()


def ollama_generate(base_url: str, payload: dict, timeout: int = 180) -> str:
    payload = {
        **payload,
        "stream": False,
        "options": {
            "temperature": payload.get("options", {}).get("temperature", 0.2),
            "num_predict": payload.get("options", {}).get("num_predict", 80),
        },
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("response", "").strip()


def clean_commentary(text: str, max_chars: int = 90, max_sentences: int = 1) -> str:
    cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    for marker in ("日本語実況：", "日本語実況:", "実況文：", "実況文:", "実況：", "実況:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[1].strip()
    cleaned = cleaned.strip("「」\"' []")
    if max_sentences > 0:
        sentence_count = 0
        end_index = None
        for index, char in enumerate(cleaned):
            if char in ("。", "！", "!"):
                sentence_count += 1
                if sentence_count >= max_sentences:
                    end_index = index + 1
                    break
        if end_index is not None:
            cleaned = cleaned[:end_index].strip()
    cleaned = cleaned[:max_chars].rstrip(" 、,")
    if cleaned and cleaned[-1] not in ("。", "！", "!", "？", "?"):
        cleaned += "。"
    return cleaned


def parse_jsonish(text: str) -> dict:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def commentary_needs_retry(text: str) -> bool:
    lowered = text.lower()
    banned = (
        "visual",
        "vehicle",
        "json",
        "画像タグ",
        "プロンプト",
        "出力",
        "英語",
        "マイル",
        "mph",
        "checkered",
        "crane",
        "実況文",
        "説明",
        "自然に取り入れ",
        "車速",
        "ではなく",
        "こと。",
        "景色:",
        "景色：",
        "景色が",
        "[",
        "]",
    )
    return not text.strip() or any(word in lowered for word in banned)


def commentary_repeat_key(text: str) -> str:
    return (
        text.replace("。", "")
        .replace("、", "")
        .replace("！", "")
        .replace("!", "")
        .replace("？", "")
        .replace("?", "")
        .strip()
    )


def commentary_is_repeat(text: str, previous: str) -> bool:
    if not previous:
        return False
    return commentary_repeat_key(text) == commentary_repeat_key(previous)


def commentary_repeats_any(text: str, recent: list[str]) -> bool:
    key = commentary_repeat_key(text)
    return bool(key) and any(key == commentary_repeat_key(item) for item in recent)


def summarize_vehicle_history(history: list[dict], window_sec: float = 6.0) -> dict:
    if not history:
        return {}
    end_time = float(history[-1].get("wall_time_sec", time.time()) or time.time())
    rows = [row for row in history if end_time - float(row.get("wall_time_sec", end_time) or end_time) <= window_sec]
    rows = rows or history[-1:]

    def values(path: tuple[str, ...]) -> list[float]:
        result = []
        for row in rows:
            value = row
            for key in path:
                value = value.get(key, {}) if isinstance(value, dict) else {}
            if isinstance(value, (int, float)):
                result.append(float(value))
        return result

    speed = values(("ego", "speed_kmh"))
    steering = values(("control_cmd", "steering_tire_angle_rad"))
    accel = values(("acceleration", "actual_accel_mps2"))
    target_speed = values(("control_cmd", "target_speed_kmh"))

    summary = {
        "window_sec": round(end_time - float(rows[0].get("wall_time_sec", end_time) or end_time), 2),
        "samples": len(rows),
    }
    if speed:
        summary["speed_kmh_start"] = round(speed[0], 2)
        summary["speed_kmh_end"] = round(speed[-1], 2)
        summary["speed_delta_kmh"] = round(speed[-1] - speed[0], 2)
        summary["speed_kmh_min"] = round(min(speed), 2)
        summary["speed_kmh_max"] = round(max(speed), 2)
    if target_speed:
        summary["target_speed_kmh_end"] = round(target_speed[-1], 2)
    if steering:
        summary["steering_start"] = round(steering[0], 3)
        summary["steering_end"] = round(steering[-1], 3)
        summary["steering_abs_max"] = round(max(abs(value) for value in steering), 3)
        if abs(steering[-1]) > 0.18:
            summary["turning"] = "left" if steering[-1] > 0 else "right"
        elif max(abs(value) for value in steering) > 0.25:
            summary["turning"] = "returning_to_straight"
        else:
            summary["turning"] = "straight_or_gentle"
    if accel:
        summary["accel_start"] = round(accel[0], 3)
        summary["accel_end"] = round(accel[-1], 3)
        summary["accel_min"] = round(min(accel), 3)
        summary["accel_max"] = round(max(accel), 3)
        if accel[-1] > 0.12:
            summary["motion_trend"] = "accelerating"
        elif accel[-1] < -0.18:
            summary["motion_trend"] = "decelerating"
        elif speed and speed[-1] - speed[0] > 2.0:
            summary["motion_trend"] = "speeding_up"
        elif speed and speed[-1] - speed[0] < -2.0:
            summary["motion_trend"] = "slowing_down"
        else:
            summary["motion_trend"] = "steady"
    return summary


def template_commentary(visual_description: str, vehicle_data: dict, style: str = "normal", event_type: str = "") -> str:
    visual = visual_description.lower()
    ego = vehicle_data.get("ego", {})
    control = vehicle_data.get("control_cmd", {})
    accel = vehicle_data.get("acceleration", {})
    speed = float(ego.get("speed_kmh", 0.0) or 0.0)
    speed_error = float(vehicle_data.get("speed_tracking_error_kmh", 0.0) or 0.0)
    actual_accel = float(accel.get("actual_accel_mps2", 0.0) or 0.0)
    steering = float(control.get("steering_tire_angle_rad", 0.0) or 0.0)

    if "left" in visual:
        direction = "左カーブ"
    elif "right" in visual:
        direction = "右カーブ"
    elif "curve" in visual or "curvy" in visual:
        direction = "カーブ"
    elif "straight" in visual:
        direction = "直線"
    elif steering > 0.18:
        direction = "左カーブ"
    elif steering < -0.18:
        direction = "右カーブ"
    else:
        direction = "コース"

    if style == "short":
        if speed < 1.0:
            return "発進待ちです。"
        if actual_accel < -0.3:
            if abs(steering) > 0.25 or "curve" in visual or "left" in visual or "right" in visual:
                return f"{direction}へ減速。"
            return "減速中です。"
        if abs(steering) > 0.3:
            turn = "左" if steering > 0 else "右"
            return f"{turn}へ旋回中。"
        if speed_error > 8.0 and actual_accel > 0.05:
            return "加速中です。"
        if actual_accel > 0.15:
            return "じわり加速。"
        if "barrier" in visual or "blue" in visual:
            return "バリア沿いです。"
        return "安定走行中。"

    if style in ("event", "event_fast"):
        if event_type == "start":
            return "走行開始です。前方を確認しながら発進します。"
        if event_type == "accel_start":
            return "前方が開けました。目標速度へ向けて加速します。"
        if event_type == "brake_start":
            if abs(steering) > 0.2 or "curve" in visual or "left" in visual or "right" in visual:
                return f"{direction}に向けて減速します。速度を落としてラインを整えます。"
            return "減速に入りました。車両姿勢を整えています。"
        if event_type == "turn_start":
            turn = "左" if steering > 0 else "右"
            return f"{turn}方向へ旋回を始めます。ステアリングを入れてラインを作ります。"
        if event_type == "turn_end":
            return "ステアリングが戻りました。直線へ向けて速度を乗せます。"
        if event_type == "target_speed_reached":
            return "目標速度に近づきました。ペースを保って走行します。"
        if event_type == "stuck":
            return "目標速度に対して伸びが鈍いです。前方状況を見ながら調整しています。"
        if event_type == "barrier_curve":
            return f"{direction}が見えています。バリア沿いにラインを合わせます。"
        if speed_error > 8.0 and actual_accel > 0.05:
            return "前方クリアです。目標速度へ向けて加速を続けます。"
        if actual_accel < -0.3:
            return "減速しています。次のラインへ車両を整えます。"
        return "走行状態が変わりました。車両の動きを確認します。"

    if style == "scenery":
        barrier_visible = "barrier" in visual and "no visible barrier" not in visual and "no barriers" not in visual
        if barrier_visible:
            return "コース脇には青白いバリアが続いています。"
        if "crane" in visual:
            return "上空にはクレーンが見えます。イベント会場の中を走っている雰囲気です。"
        if "cityscape" in visual or "building" in visual or "structure" in visual:
            return "周囲には大きな建物や街並みが見えます。都市型コースらしい景色です。"
        if "tree" in visual or "grass" in visual or "greenery" in visual or "green" in visual:
            return "コース周辺に緑が見えます。青空の下を抜けていきます。"
        if "road" in visual or "track" in visual:
            return "前方にはコースが続いています。"
        if "sky" in visual or "cloud" in visual:
            return "空が広く見えます。開けたコースを走っています。"
        return "車載カメラの上側から、周囲の景色を確認しています。"

    if speed < 1.0:
        return "スタート直後、前方のコースを確認しています。"
    if actual_accel < -0.3:
        if abs(steering) > 0.25 or "curve" in visual or "left" in visual or "right" in visual:
            return f"減速しながら{direction}に備えています。"
        return "速度を落として姿勢を整えています。"
    if abs(steering) > 0.3:
        turn = "左" if steering > 0 else "右"
        return f"{turn}へ舵を入れ、ラインを整えています。"
    if speed_error > 8.0 and actual_accel > 0.05:
        return "前方クリア、目標速度へ向けて加速しています。"
    if actual_accel > 0.15:
        return f"{direction}を抜けながら、じわりと加速しています。"
    if "barrier" in visual or "blue" in visual:
        return "青白いバリア沿いに安定して走行しています。"
    return "コースを安定して走行中です。"


def synthesize_voicevox(base_url: str, speaker: int, text: str, output_path: Path, speed_scale: float) -> None:
    query_url = f"{base_url.rstrip('/')}/audio_query?" + urllib.parse.urlencode(
        {"text": text, "speaker": speaker}
    )
    with urllib.request.urlopen(urllib.request.Request(query_url, method="POST"), timeout=20) as response:
        query = json.loads(response.read().decode("utf-8"))
    query["speedScale"] = speed_scale
    query["prePhonemeLength"] = 0.02
    query["postPhonemeLength"] = 0.04
    synth_url = f"{base_url.rstrip('/')}/synthesis?" + urllib.parse.urlencode({"speaker": speaker})
    req = urllib.request.Request(
        synth_url,
        data=json.dumps(query).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=40) as response:
        output_path.write_bytes(response.read())


class RealtimeVLMCommentary(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("realtime_vlm_commentary")
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.busy = False
        self.latest_image = None
        self.latest_image_stamp = None
        self.latest_odom = None
        self.latest_control = None
        self.latest_accel = None
        self.index = 0
        self.last_event_data = None
        self.last_spoken_wall_time = 0.0
        self.last_generated_image_hash = ""
        self.last_commentary = ""
        self.last_visual_description = ""
        self.recent_commentaries = []
        self.vehicle_history = []
        self.previous_vlm_image_b64 = ""
        self.previous_vlm_image = None

        self.output_dir = Path(args.output_dir)
        self.frame_dir = self.output_dir / "frames"
        self.audio_dir = self.output_dir / "voicevox"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        if args.voicevox_url:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "realtime_commentary.jsonl"

        self.create_subscription(Image, args.image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, 20)
        if AckermannControlCommand is not None:
            self.create_subscription(AckermannControlCommand, args.control_topic, self.on_control, 20)
        self.create_subscription(AccelWithCovarianceStamped, args.accel_topic, self.on_accel, 20)
        self.create_timer(args.interval_sec, self.on_timer)
        self.get_logger().info(
            f"started interval={args.interval_sec}s preprocess={args.preprocess} output={self.output_dir}"
        )

    def on_image(self, msg: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"image conversion failed: {exc}")
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def on_odom(self, msg: Odometry) -> None:
        with self.lock:
            self.latest_odom = msg
            self.append_vehicle_history_locked()

    def on_control(self, msg) -> None:
        with self.lock:
            self.latest_control = msg
            self.append_vehicle_history_locked()

    def on_accel(self, msg: AccelWithCovarianceStamped) -> None:
        with self.lock:
            self.latest_accel = msg
            self.append_vehicle_history_locked()

    def append_vehicle_history_locked(self) -> None:
        sample = self.vehicle_data(None, self.latest_odom, self.latest_control, self.latest_accel)
        sample["image_stamp_sec"] = None
        self.vehicle_history.append(sample)
        cutoff = time.time() - 8.0
        self.vehicle_history = [
            row for row in self.vehicle_history[-300:] if float(row.get("wall_time_sec", 0.0) or 0.0) >= cutoff
        ]

    def on_timer(self) -> None:
        if self.busy:
            return
        with self.lock:
            if self.latest_image is None:
                self.get_logger().info("waiting for image")
                return
            image = self.latest_image.copy()
            image_stamp = self.latest_image_stamp
            odom = self.latest_odom
            control = self.latest_control
            accel = self.latest_accel
            vehicle_history = list(self.vehicle_history)
        self.busy = True
        threading.Thread(
            target=self.generate_once,
            args=(image, image_stamp, odom, control, accel, vehicle_history),
            daemon=True,
        ).start()

    def vehicle_data(self, image_stamp, odom, control, accel) -> dict:
        data = {
            "image_stamp_sec": image_stamp,
            "wall_time_sec": time.time(),
            "preprocess": self.args.preprocess,
            "controller": "mpc_or_current_control_topic",
        }
        if odom is not None:
            speed_mps = float(odom.twist.twist.linear.x)
            data["ego"] = {
                "speed_mps": round(speed_mps, 3),
                "speed_kmh": round(speed_mps * 3.6, 3),
                "pose_x": round(float(odom.pose.pose.position.x), 3),
                "pose_y": round(float(odom.pose.pose.position.y), 3),
            }
        if control is not None:
            target_speed = float(getattr(control.longitudinal, "speed", 0.0))
            data["control_cmd"] = {
                "target_speed_mps": round(target_speed, 3),
                "target_speed_kmh": round(target_speed * 3.6, 3),
                "target_accel_mps2": round(float(getattr(control.longitudinal, "acceleration", 0.0)), 3),
                "steering_tire_angle_rad": round(float(getattr(control.lateral, "steering_tire_angle", 0.0)), 3),
            }
        if accel is not None:
            data["acceleration"] = {
                "actual_accel_mps2": round(float(accel.accel.accel.linear.x), 3),
                "actual_yaw_accel_mps2": round(float(accel.accel.accel.angular.z), 3),
            }
        if data.get("ego") and data.get("control_cmd"):
            data["speed_tracking_error_kmh"] = round(
                data["control_cmd"]["target_speed_kmh"] - data["ego"]["speed_kmh"],
                3,
            )
        return data

    def detect_event(self, data: dict) -> tuple[str, str]:
        now = float(data.get("wall_time_sec", time.time()))
        ego = data.get("ego", {})
        control = data.get("control_cmd", {})
        accel = data.get("acceleration", {})
        speed = float(ego.get("speed_kmh", 0.0) or 0.0)
        speed_error = float(data.get("speed_tracking_error_kmh", 0.0) or 0.0)
        actual_accel = float(accel.get("actual_accel_mps2", 0.0) or 0.0)
        steering = float(control.get("steering_tire_angle_rad", 0.0) or 0.0)

        prev = self.last_event_data or {}
        prev_speed_error = float(prev.get("speed_error", 0.0) or 0.0)
        prev_accel = float(prev.get("actual_accel", 0.0) or 0.0)
        prev_steering = float(prev.get("steering", 0.0) or 0.0)
        prev_speed = float(prev.get("speed", 0.0) or 0.0)

        state = {
            "speed": speed,
            "speed_error": speed_error,
            "actual_accel": actual_accel,
            "steering": steering,
        }

        event = ""
        reason = ""
        if not prev:
            event = "start"
            reason = "first_sample"
        elif speed > 1.0 and prev_speed < 1.0:
            event = "start"
            reason = "speed_crossed_1kmh"
        elif actual_accel > 0.18 and prev_accel <= 0.05 and speed_error > 5.0:
            event = "accel_start"
            reason = "accel_positive_crossing"
        elif actual_accel < -0.3 and prev_accel >= -0.1:
            event = "brake_start"
            reason = "accel_negative_crossing"
        elif abs(steering) > 0.25 and abs(prev_steering) <= 0.16:
            event = "turn_start"
            reason = "steering_enter"
        elif abs(prev_steering) > 0.25 and abs(steering) <= 0.12:
            event = "turn_end"
            reason = "steering_exit"
        elif abs(speed_error) < 3.0 and abs(prev_speed_error) >= 8.0 and speed > 5.0:
            event = "target_speed_reached"
            reason = "speed_error_converged"
        elif speed_error > 12.0 and speed < 5.0 and actual_accel < 0.05:
            event = "stuck"
            reason = "large_speed_error_low_accel"

        self.last_event_data = state
        if not event:
            return "", "no_event"
        if now - self.last_spoken_wall_time < self.args.min_speak_interval_sec:
            return "", f"min_interval:{event}:{reason}"
        self.last_spoken_wall_time = now
        return event, reason

    def generate_once(self, image, image_stamp, odom, control, accel, vehicle_history) -> None:
        started = time.perf_counter()
        generation_wall_time = time.time()
        try:
            data = self.vehicle_data(image_stamp, odom, control, accel)
            event_type = ""
            event_reason = ""
            if self.args.commentary_trigger == "event":
                event_type, event_reason = self.detect_event(data)
                if not event_type:
                    self.get_logger().info(
                        json.dumps(
                            {
                                "skipped": True,
                                "reason": event_reason,
                                "vehicle_data": data,
                            },
                            ensure_ascii=False,
                        )
                    )
                    return

            processed = preprocess_image(image, self.args.preprocess)
            image_brightness = float(processed.mean())
            image_hash = hashlib.md5(processed.tobytes()).hexdigest()
            if image_brightness < self.args.min_image_brightness:
                self.get_logger().info(
                    json.dumps(
                        {
                            "skipped": True,
                            "reason": "dark_frame",
                            "image_stamp_sec": image_stamp,
                            "image_brightness": round(image_brightness, 3),
                        },
                        ensure_ascii=False,
                    )
                )
                return
            if image_hash == self.last_generated_image_hash:
                self.get_logger().info(
                    json.dumps(
                        {
                            "skipped": True,
                            "reason": "duplicate_frame",
                            "image_stamp_sec": image_stamp,
                            "image_brightness": round(image_brightness, 3),
                        },
                        ensure_ascii=False,
                    )
                )
                return
            self.last_generated_image_hash = image_hash
            frame_path = self.frame_dir / f"frame_{self.index:04d}.jpg"
            cv2.imwrite(str(frame_path), processed, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            current_image_b64 = encode_jpeg_b64(processed)
            previous_image = self.previous_vlm_image.copy() if self.previous_vlm_image is not None else None
            visual = ""
            visual_tags = {}
            vision_sec = 0.0
            if self.args.template_style != "event_fast":
                visual_prompt = (
                    "Describe only the upper camera scenery from this vehicle camera. "
                    "Mention sky, barriers, buildings, trees, or distant course features if visible. "
                    "Do not mention UI, screenshots, RViz, or simulator. "
                    "One short English sentence."
                )
                if self.args.template_style != "scenery":
                    visual_prompt = (
                        "Describe only the road scene from this vehicle camera. "
                        "Mention road direction or barriers if visible. "
                        "Do not mention UI, screenshots, RViz, or simulator. "
                        "One short English sentence."
                    )
                if "moondream" in self.args.vision_model.lower():
                    if previous_image is not None:
                        visual_prompt = (
                            "This is one comparison image from a vehicle camera. The left half is earlier and the right half is current. "
                            "Describe what visibly changed in one short English sentence. "
                            "Mention road direction, barriers, buildings, sky, or course if changed."
                        )
                    else:
                        visual_prompt = (
                            "Describe this driving scene briefly. "
                            "Mention visible road, sky, barriers, buildings, or course."
                        )
                if self.args.template_style == "vlm_tags":
                    visual_prompt = (
                        "You are labeling a vehicle-mounted camera image for autonomous driving commentary. "
                        "Return only minified JSON with these keys: "
                        "road_direction, curve, barrier, sky, buildings, scene, confidence. "
                        "Allowed road_direction values: left, right, straight, unknown. "
                        "Allowed curve values: none, gentle, sharp, unknown. "
                        "barrier, sky, buildings are booleans. "
                        "scene is one short snake_case label. "
                        "No markdown. No explanation."
                    )
                if "moondream" in self.args.vision_model.lower() and previous_image is not None:
                    comparison_image = cv2.hconcat([previous_image, processed])
                    image_payload = [encode_jpeg_b64(comparison_image)]
                else:
                    image_payload = [current_image_b64]
                visual_payload = {
                    "model": self.args.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": visual_prompt,
                            "images": image_payload,
                        }
                    ],
                    "keep_alive": self.args.ollama_keep_alive,
                    "options": {
                        "num_predict": 120 if self.args.template_style == "vlm_tags" else 50,
                        "temperature": 0.0 if self.args.template_style == "vlm_tags" else 0.1,
                    },
                }
                t0 = time.perf_counter()
                visual = ollama_chat(self.args.ollama_base_url, visual_payload)
                if "moondream" in self.args.vision_model.lower() and not visual.strip():
                    fallback_prompt = (
                        "Caption the current driving camera image in one short sentence."
                    )
                    fallback_chat_payload = {
                        **visual_payload,
                        "messages": [
                            {
                                "role": "user",
                                "content": fallback_prompt,
                                "images": [current_image_b64],
                            }
                        ],
                        "options": {"num_predict": 50, "temperature": 0.0},
                    }
                    visual = ollama_chat(self.args.ollama_base_url, fallback_chat_payload)
                    if not visual.strip():
                        fallback_generate_payload = {
                            "model": self.args.vision_model,
                            "prompt": fallback_prompt,
                            "images": [current_image_b64],
                            "keep_alive": self.args.ollama_keep_alive,
                            "options": {"num_predict": 50, "temperature": 0.0},
                        }
                        visual = ollama_generate(self.args.ollama_base_url, fallback_generate_payload)
                vision_sec = time.perf_counter() - t0
                if self.args.template_style == "vlm_tags":
                    visual_tags = parse_jsonish(visual)
                self.previous_vlm_image_b64 = current_image_b64
                self.previous_vlm_image = processed.copy()

            if self.args.commentary_mode == "template":
                t1 = time.perf_counter()
                commentary = template_commentary(visual, data, self.args.template_style, event_type)
                text_sec = time.perf_counter() - t1
            else:
                previous_commentary = self.last_commentary
                previous_visual = self.last_visual_description
                recent_commentaries = list(self.recent_commentaries[-6:])
                vehicle_summary = summarize_vehicle_history(vehicle_history)
                commentary_types = [
                    "visual_change",
                    "vehicle_motion",
                    "course_shape",
                    "scene_detail",
                    "sport_commentary",
                ]
                commentary_type = commentary_types[self.index % len(commentary_types)]
                if self.args.template_style == "vlm_tags":
                    ego = data.get("ego", {})
                    control = data.get("control_cmd", {})
                    accel = data.get("acceleration", {})
                    compact_data = {
                        "speed_kmh": ego.get("speed_kmh"),
                        "target_speed_kmh": control.get("target_speed_kmh"),
                        "actual_accel_mps2": accel.get("actual_accel_mps2"),
                        "steering_rad": control.get("steering_tire_angle_rad"),
                    }
                    prompt = f"""
あなたは車載カメラ映像の短い実況者です。
次の visual_tags と vehicle だけを使い、日本語1文を作ってください。
タグの説明文は禁止。速度や数値の羅列は禁止。英語は禁止。
20〜35文字。句点で終える。
減速中なら加速と言わない。

良い例:
空の下、直線コースを伸びていきます。
建物を横目に、右カーブへ入ります。
バリア沿いに速度を落として曲がります。

previous_commentary:
{previous_commentary}

recent_commentaries:
{json.dumps(recent_commentaries, ensure_ascii=False)}

commentary_type:
{commentary_type}

vehicle_summary:
{json.dumps(vehicle_summary, ensure_ascii=False)}

visual_tags:
{json.dumps(visual_tags, ensure_ascii=False)}

vehicle:
{json.dumps(compact_data, ensure_ascii=False)}
""".strip()
                    text_options = {"num_predict": 50, "temperature": 0.25}
                    system_prompt = "日本語の実況1文だけを返す。説明、前置き、箇条書きは禁止。"
                else:
                    prompt = f"""
あなたは自動運転カートの実況者です。
使ってよい情報は visual_description と vehicle_data JSON だけです。
AWSIM外部視点、RViz、UIには触れないでください。
actual_accel_mps2 が -0.3 未満なら「加速中」とは言わないでください。
出力は日本語の実況1文だけ。20〜40文字。
英語、数値、単位、JSON、説明、前置きは禁止。
景色、コース、バリア、建物のどれかを入れてください。
「加速中」「走行中」のような汎用文だけは禁止です。
recent_commentaries と同じ文、同じ言い回しは禁止です。
前回と同じ景色なら、建物、看板、コース形状、空、路面、バリアのうち別の要素を拾ってください。
commentary_type に合わせて話題を変えてください。
- visual_change: 前回画像から変わった見た目を話す
- vehicle_motion: 加速、減速、旋回、安定など車両挙動を話す
- course_shape: 直線、カーブ、縁石、バリアなどコース形状を話す
- scene_detail: 建物、看板、空、工事中の景色などを話す
- sport_commentary: 少し実況者らしく、ただし大げさにしすぎない

良い例:
ビルの間から、コースが右へ続きます。
建設中の景色を横目に、ラインを整えます。
看板のある街中コースを、落ち着いて進みます。
街の看板を横目に、コースを抜けます。

previous_commentary:
{previous_commentary}

recent_commentaries:
{json.dumps(recent_commentaries, ensure_ascii=False)}

previous_visual_description:
{previous_visual}

commentary_type:
{commentary_type}

visual_change:
{visual}

vehicle_summary:
{json.dumps(vehicle_summary, ensure_ascii=False)}

vehicle_data:
{json.dumps(data, ensure_ascii=False)}
""".strip()
                    text_options = {"num_predict": 70, "temperature": 0.3}
                    system_prompt = "Return only one concise Japanese commentary sentence."
                text_payload = {
                    "model": self.args.text_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "think": False,
                    "keep_alive": self.args.ollama_keep_alive,
                    "options": text_options,
                }
                t1 = time.perf_counter()
                raw_commentary = ollama_chat(self.args.ollama_base_url, text_payload)
                commentary = clean_commentary(raw_commentary, self.args.max_commentary_chars, 1)
                if commentary_needs_retry(commentary) or commentary_repeats_any(commentary, recent_commentaries):
                    retry_payload = {
                        **text_payload,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "日本語の短い実況1文だけを返す。英語、数値、単位、JSON、説明、"
                                    "プロンプト文、前置きは禁止。"
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    "次を自然な日本語実況に直してください。20〜35文字。"
                                    "景色、コース、バリア、建物、看板、空のどれかを入れる。"
                                    "履歴と同じ文、同じ言い回しは禁止。\n\n"
                                    f"最近の実況: {json.dumps(recent_commentaries, ensure_ascii=False)}\n"
                                    f"実況タイプ: {commentary_type}\n"
                                    f"VLM変化: {visual}\n"
                                    f"車両履歴: {json.dumps(vehicle_summary, ensure_ascii=False)}\n"
                                    f"車両状態: {json.dumps(data, ensure_ascii=False)}"
                                ),
                            },
                        ],
                        "options": {"num_predict": 45, "temperature": 0.2},
                    }
                    commentary = clean_commentary(
                        ollama_chat(self.args.ollama_base_url, retry_payload),
                        self.args.max_commentary_chars,
                        1,
                    )
                if commentary_needs_retry(commentary):
                    commentary = template_commentary(visual, data, "normal", event_type)
                if commentary_repeats_any(commentary, recent_commentaries):
                    commentary = template_commentary(visual, data, "normal", event_type)
                text_sec = time.perf_counter() - t1

            template_sentences = 2 if self.args.template_style in ("event", "scenery") else 1
            commentary = clean_commentary(commentary, self.args.max_commentary_chars, template_sentences)

            audio_path = None
            tts_sec = None
            if self.args.voicevox_url:
                audio_path = self.audio_dir / f"line_{self.index:04d}.wav"
                t2 = time.perf_counter()
                synthesize_voicevox(
                    self.args.voicevox_url,
                    self.args.voicevox_speaker,
                    commentary,
                    audio_path,
                    self.args.voicevox_speed_scale,
                )
                tts_sec = time.perf_counter() - t2
            audio_ready_wall_time = time.time()

            record = {
                "index": self.index,
                "image_stamp_sec": image_stamp,
                "generation_wall_time_sec": generation_wall_time,
                "audio_ready_wall_time_sec": audio_ready_wall_time,
                "frame": str(frame_path),
                "image_brightness": round(image_brightness, 3),
                "visual_description": visual,
                "visual_tags": visual_tags,
                "vehicle_data": data,
                "commentary": commentary,
                "commentary_mode": self.args.commentary_mode,
                "commentary_trigger": self.args.commentary_trigger,
                "event_type": event_type or None,
                "event_reason": event_reason or None,
                "commentary_type": commentary_type if self.args.commentary_mode != "template" else None,
                "vehicle_summary": vehicle_summary if self.args.commentary_mode != "template" else None,
                "template_style": self.args.template_style,
                "audio": str(audio_path) if audio_path else None,
                "voicevox_speed_scale": self.args.voicevox_speed_scale if audio_path else None,
                "latency_sec": {
                    "vision": round(vision_sec, 3),
                    "text": round(text_sec, 3),
                    "tts": round(tts_sec, 3) if tts_sec is not None else None,
                    "total": round(time.perf_counter() - started, 3),
                },
            }
            with self.jsonl_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.get_logger().info(json.dumps(record, ensure_ascii=False))
            self.last_commentary = commentary
            self.last_visual_description = visual
            self.recent_commentaries.append(commentary)
            self.recent_commentaries = self.recent_commentaries[-6:]
            self.index += 1
        except Exception as exc:
            self.get_logger().error(f"generation failed: {exc}")
        finally:
            self.busy = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-topic", default="/sensing/camera/image_raw")
    parser.add_argument("--odom-topic", default="/localization/kinematic_state")
    parser.add_argument("--control-topic", default="/control/command/control_cmd")
    parser.add_argument("--accel-topic", default="/localization/acceleration")
    parser.add_argument("--output-dir", default="/tmp/realtime_vlm_commentary")
    parser.add_argument("--interval-sec", type=float, default=3.0)
    parser.add_argument("--preprocess", default="lower_80x40", choices=["full", "full_320x180", "full_256x144", "upper_half", "upper_320x160", "upper_160x80", "lower_half", "lower_160x80", "lower_80x40"])
    parser.add_argument("--min-image-brightness", type=float, default=float(os.getenv("MIN_IMAGE_BRIGHTNESS", "1.0")))
    parser.add_argument("--commentary-mode", default="llm", choices=["llm", "template"])
    parser.add_argument("--commentary-trigger", default=os.getenv("COMMENTARY_TRIGGER", "interval"), choices=["interval", "event"])
    parser.add_argument("--min-speak-interval-sec", type=float, default=float(os.getenv("MIN_SPEAK_INTERVAL_SEC", "4.0")))
    parser.add_argument("--template-style", default=os.getenv("TEMPLATE_STYLE", "normal"), choices=["normal", "short", "event", "event_fast", "scenery", "vlm_tags"])
    parser.add_argument("--max-commentary-chars", type=int, default=int(os.getenv("MAX_COMMENTARY_CHARS", "45")))
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-keep-alive", default=os.getenv("OLLAMA_KEEP_ALIVE", "30m"))
    parser.add_argument("--vision-model", default=os.getenv("OLLAMA_VISION_MODEL", "llava:7b"))
    parser.add_argument("--text-model", default=os.getenv("OLLAMA_TEXT_MODEL", "qwen3:8b"))
    parser.add_argument("--voicevox-url", default=os.getenv("VOICEVOX_URL", ""))
    parser.add_argument("--voicevox-speaker", type=int, default=int(os.getenv("VOICEVOX_SPEAKER", "3")))
    parser.add_argument("--voicevox-speed-scale", type=float, default=float(os.getenv("VOICEVOX_SPEED_SCALE", "1.08")))
    args = parser.parse_args()

    rclpy.init()
    node = RealtimeVLMCommentary(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
