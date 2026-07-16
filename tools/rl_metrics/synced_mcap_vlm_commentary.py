#!/usr/bin/env python3
"""Generate VLM commentary synchronized with an AI Challenge camera video and mcap data."""

from __future__ import annotations

import argparse
import base64
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import urllib.request

import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


CONTROL_TOPIC = "/control/command/control_cmd"
ODOM_TOPIC = "/localization/kinematic_state"
ACCEL_TOPIC = "/localization/acceleration"


def parse_times(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def ns_to_sec(ns: int) -> float:
    return ns / 1_000_000_000.0


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def clean_commentary(text: str) -> str:
    cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    for marker in ("実況文：", "実況文:", "実況：", "実況:"):
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[1].strip()
    for prefix in ("こんにちは！", "こんにちは。", "こんにちは"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.strip("「」\"' ")
    for sep in ("。", "！", "!"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip() + "。"
            break
    return cleaned[:90]


def ollama_chat(base_url: str, payload: dict, timeout: int = 180) -> str:
    payload = {
        **payload,
        "stream": False,
        "options": {
            "temperature": payload.get("options", {}).get("temperature", 0.2),
            "num_predict": payload.get("options", {}).get("num_predict", 128),
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


def describe_camera_image(base_url: str, vision_model: str, image_path: Path) -> str:
    payload = {
        "model": vision_model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Describe only the road scene visible from this vehicle-mounted camera. "
                    "Focus on track direction, barriers, open road, and traffic. "
                    "Do not mention windows, UI, screenshots, developers, RViz, or simulators. "
                    "Use one short English sentence."
                ),
                "images": [encode_image(image_path)],
            }
        ],
        "options": {"num_predict": 80},
    }
    return ollama_chat(base_url, payload)


def generate_commentary(base_url: str, text_model: str, visual_description: str, vehicle_data: dict) -> str:
    prompt = f"""
あなたは自動運転カートの実況者です。
使ってよい情報は visual_description と vehicle_data JSON だけです。
AWSIM外部視点、RViz画面、開発者画面、UI、スクリーンショットには触れないでください。
速度や加減速が特徴的なら自然に触れてください。ただし毎回数字を読むだけの実況にしないでください。
加減速を表現するときは control_cmd.target_accel_mps2 より acceleration.actual_accel_mps2 を優先してください。
actual_accel_mps2 が -0.3 未満なら「加速中」とは言わず、減速、姿勢を整える、旋回に備える等と表現してください。
出力は日本語の実況文1文だけ。挨拶、前置き、箇条書きは禁止。55文字以内。

visual_description:
{visual_description}

vehicle_data:
{json.dumps(vehicle_data, ensure_ascii=False, indent=2)}
""".strip()
    payload = {
        "model": text_model,
        "messages": [
            {
                "role": "system",
                "content": "Return only one concise Japanese commentary sentence. No thinking text.",
            },
            {"role": "user", "content": prompt},
        ],
        "think": False,
        "options": {"num_predict": 90, "temperature": 0.3},
    }
    return clean_commentary(ollama_chat(base_url, payload))


def open_bag(bag_path: Path) -> rosbag2_py.SequentialReader:
    storage_id = "mcap" if bag_path.suffix == ".mcap" else "sqlite3"
    uri = str(bag_path if bag_path.suffix != ".mcap" else bag_path)
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=uri, storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def read_bag_samples(bag_path: Path) -> tuple[int, dict[str, list[dict]]]:
    reader = open_bag(bag_path)
    topic_types = {
        topic_metadata.name: topic_metadata.type
        for topic_metadata in reader.get_all_topics_and_types()
    }
    msg_types = {topic: get_message(type_name) for topic, type_name in topic_types.items()}
    samples: dict[str, list[dict]] = {ODOM_TOPIC: [], CONTROL_TOPIC: [], ACCEL_TOPIC: []}
    bag_start_ns: int | None = None

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if bag_start_ns is None or timestamp_ns < bag_start_ns:
            bag_start_ns = timestamp_ns
        if topic not in samples:
            continue
        msg = deserialize_message(data, msg_types[topic])
        item = {"timestamp_ns": timestamp_ns}
        if topic == ODOM_TOPIC:
            speed_mps = float(msg.twist.twist.linear.x)
            item.update(
                {
                    "speed_mps": speed_mps,
                    "speed_kmh": speed_mps * 3.6,
                    "pose_x": float(msg.pose.pose.position.x),
                    "pose_y": float(msg.pose.pose.position.y),
                }
            )
        elif topic == CONTROL_TOPIC:
            item.update(
                {
                    "target_speed_mps": float(getattr(msg.longitudinal, "speed", math.nan)),
                    "target_speed_kmh": float(getattr(msg.longitudinal, "speed", math.nan)) * 3.6,
                    "target_accel_mps2": float(getattr(msg.longitudinal, "acceleration", math.nan)),
                    "steering_tire_angle_rad": float(getattr(msg.lateral, "steering_tire_angle", math.nan)),
                }
            )
        elif topic == ACCEL_TOPIC:
            item.update(
                {
                    "actual_accel_mps2": float(msg.accel.accel.linear.x),
                    "actual_yaw_accel_mps2": float(msg.accel.accel.angular.z),
                }
            )
        samples[topic].append(item)

    if bag_start_ns is None:
        raise RuntimeError(f"empty bag: {bag_path}")
    return bag_start_ns, samples


def nearest_sample(samples: list[dict], target_ns: int) -> dict | None:
    if not samples:
        return None
    timestamps = [item["timestamp_ns"] for item in samples]
    pos = bisect_left(timestamps, target_ns)
    candidates = []
    if pos < len(samples):
        candidates.append(samples[pos])
    if pos > 0:
        candidates.append(samples[pos - 1])
    best = min(candidates, key=lambda item: abs(item["timestamp_ns"] - target_ns))
    result = dict(best)
    result["delta_sec"] = ns_to_sec(best["timestamp_ns"] - target_ns)
    result["bag_elapsed_sec"] = ns_to_sec(best["timestamp_ns"] - samples[0]["timestamp_ns"])
    return result


def round_floats(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 3)
    if isinstance(value, dict):
        return {k: round_floats(v) for k, v in value.items() if k != "timestamp_ns"}
    if isinstance(value, list):
        return [round_floats(v) for v in value]
    return value


def find_capture_start_offset(video_path: Path, bag_start_ns: int, explicit_offset: float | None) -> float:
    if explicit_offset is not None:
        return explicit_offset
    candidates = sorted(video_path.parent.glob("cap-*.mp4"))
    if not candidates:
        return 0.0
    match = re.search(r"cap-(\d{8})-(\d{6})\.mp4$", candidates[0].name)
    if not match:
        return 0.0
    # The capture filename is produced in the cc1 local timezone (JST), while
    # rosbag2 timestamps are Unix epoch seconds. Treat the filename as JST.
    capture_start = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    capture_start = capture_start.replace(tzinfo=timezone(timedelta(hours=9)))
    return capture_start.timestamp() - ns_to_sec(bag_start_ns)


def extract_frame(
    cap: cv2.VideoCapture,
    time_sec: float,
    preprocess: str,
    output_path: Path,
) -> None:
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame at {time_sec:.2f}s")
    if preprocess == "lower_half":
        frame = frame[frame.shape[0] // 2 :, :]
    elif preprocess == "lower_half_160x80":
        frame = frame[frame.shape[0] // 2 :, :]
        frame = cv2.resize(frame, (160, 80), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (640, 320), interpolation=cv2.INTER_NEAREST)
    elif preprocess != "full":
        raise ValueError(f"unknown preprocess: {preprocess}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def build_vehicle_data(
    video_time_sec: float,
    bag_target_elapsed_sec: float,
    samples: dict[str, list[dict]],
    target_ns: int,
) -> dict:
    odom = nearest_sample(samples[ODOM_TOPIC], target_ns)
    control = nearest_sample(samples[CONTROL_TOPIC], target_ns)
    accel = nearest_sample(samples[ACCEL_TOPIC], target_ns)
    speed = odom.get("speed_kmh") if odom else None
    target_speed = control.get("target_speed_kmh") if control else None
    speed_error = target_speed - speed if speed is not None and target_speed is not None else None
    return round_floats(
        {
            "video_time_sec": video_time_sec,
            "bag_target_elapsed_sec": bag_target_elapsed_sec,
            "controller": "mpc",
            "source_run": "mpc-camera-rosbag-20260712-231404",
            "allowed_visual_input": "vehicle-mounted camera video only",
            "ego": odom,
            "control_cmd": control,
            "acceleration": accel,
            "speed_tracking_error_kmh": speed_error,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--times", default="5,15,25,35,45,55,65,75,85,95")
    parser.add_argument("--preprocess", default="lower_half_160x80", choices=["full", "lower_half", "lower_half_160x80"])
    parser.add_argument("--bag-video-offset-sec", type=float, default=None)
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--vision-model", default=os.getenv("OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "llava:7b")))
    parser.add_argument("--text-model", default=os.getenv("OLLAMA_TEXT_MODEL", "qwen3:8b"))
    args = parser.parse_args()

    video = Path(args.video)
    bag = Path(args.bag)
    output_dir = Path(args.output_dir)

    bag_start_ns, samples = read_bag_samples(bag)
    video_offset_sec = find_capture_start_offset(video, bag_start_ns, args.bag_video_offset_sec)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else None

    records = []
    for index, time_sec in enumerate(parse_times(args.times)):
        if duration is not None and time_sec >= duration:
            continue
        frame_path = output_dir / "frames" / f"{args.preprocess}_t{time_sec:06.2f}.jpg"
        extract_frame(cap, time_sec, args.preprocess, frame_path)
        bag_target_elapsed_sec = video_offset_sec + time_sec
        target_ns = int(bag_start_ns + bag_target_elapsed_sec * 1_000_000_000)
        vehicle_data = build_vehicle_data(time_sec, bag_target_elapsed_sec, samples, target_ns)
        visual_description = describe_camera_image(args.ollama_base_url, args.vision_model, frame_path)
        commentary = generate_commentary(args.ollama_base_url, args.text_model, visual_description, vehicle_data)
        record = {
            "index": index,
            "time_sec": time_sec,
            "bag_target_elapsed_sec": round(bag_target_elapsed_sec, 3),
            "frame": str(frame_path),
            "preprocess": args.preprocess,
            "vehicle_data": vehicle_data,
            "visual_description": visual_description,
            "commentary": commentary,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "synced_vehicle_data.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record["vehicle_data"], ensure_ascii=False) + "\n")
    with (output_dir / "commentary.jsonl").open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (output_dir / "commentary.md").open("w", encoding="utf-8") as file:
        file.write("# Synced MPC camera + mcap VLM commentary\n\n")
        file.write(f"- video: `{video}`\n")
        file.write(f"- bag: `{bag}`\n")
        file.write(f"- preprocess: `{args.preprocess}`\n")
        file.write(f"- video_offset_sec: `{video_offset_sec:.3f}`\n")
        file.write(f"- vision_model: `{args.vision_model}`\n")
        file.write(f"- text_model: `{args.text_model}`\n\n")
        for record in records:
            data = record["vehicle_data"]
            speed = data.get("ego", {}).get("speed_kmh")
            target = data.get("control_cmd", {}).get("target_speed_kmh")
            accel = data.get("control_cmd", {}).get("target_accel_mps2")
            file.write(
                f"- {record['time_sec']:.1f}s "
                f"(speed={speed}km/h, target={target}km/h, accel={accel}m/s2): "
                f"{record['commentary']}\n"
            )


if __name__ == "__main__":
    main()
