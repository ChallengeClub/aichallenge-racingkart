#!/usr/bin/env python3
"""Generate sensor-grounded VLM commentary from camera frames and vehicle data."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import urllib.request

import cv2


def parse_times(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def ollama_chat(base_url: str, payload: dict) -> str:
    payload = {
        **payload,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": payload.get("options", {}).get("num_predict", 128),
        },
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
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
AWSIM外部視点、RViz画面、開発者向け画面、UI、スクリーンショットには触れないでください。
出力は日本語の実況文1文だけ。挨拶、前置き、箇条書きは禁止。45文字以内。
unknownの数値は言わないでください。

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
        "options": {"num_predict": 80},
    }
    return clean_commentary(ollama_chat(base_url, payload))


def clean_commentary(text: str) -> str:
    cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if "実況文" in cleaned and "：" in cleaned:
        cleaned = cleaned.split("：", 1)[1].strip()
    for prefix in ("こんにちは！", "こんにちは。", "こんにちは", "実況:", "実況："):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    cleaned = cleaned.strip("「」\"' ")
    for sep in ("。", "！", "!"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0].strip() + "。"
            break
    return cleaned[:80]


def extract_camera_frame(
    cap: cv2.VideoCapture,
    time_sec: float,
    crop: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError(f"failed to read frame at {time_sec:.2f}s")
    x, y, w, h = crop
    cropped = frame[y : y + h, x : x + w]
    if cropped.size == 0:
        raise RuntimeError(f"empty crop at {time_sec:.2f}s: crop={crop}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cropped)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--times", default="5,15,25,35")
    parser.add_argument("--crop", default="168,664,452,300", help="x,y,w,h for vehicle camera image")
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--vision-model", default=os.getenv("OLLAMA_VISION_MODEL", os.getenv("OLLAMA_MODEL", "llava:7b")))
    parser.add_argument("--text-model", default=os.getenv("OLLAMA_TEXT_MODEL", "qwen3:8b"))
    args = parser.parse_args()

    video = Path(args.video)
    output_dir = Path(args.output_dir)
    crop = tuple(int(v.strip()) for v in args.crop.split(","))
    if len(crop) != 4:
        raise ValueError("--crop must be x,y,w,h")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else None

    records = []
    for index, time_sec in enumerate(parse_times(args.times)):
        frame_path = output_dir / "frames" / f"camera_t{time_sec:06.2f}.jpg"
        extract_camera_frame(cap, time_sec, crop, frame_path)

        vehicle_data = {
            "time_sec": time_sec,
            "source": "cropped vehicle camera viewer subscribed to /sensing/camera/image_raw",
            "video": str(video),
            "video_duration_sec": duration,
            "controller": {
                "name": "pilot_net_controller",
                "mode": "fixed",
                "note": "from reference.log: OutputDim=2, Mode=fixed",
            },
            "trajectory_source": "raceline_awsim_2025blend_blend30.csv",
            "ego_speed_kmh": "unknown_in_this_recording",
            "v2x_distance_m": "unknown_in_this_recording",
            "penalty": "unknown_in_this_recording",
            "allowed_visual_input": "vehicle camera image only",
        }

        visual_description = describe_camera_image(args.ollama_base_url, args.vision_model, frame_path)
        commentary = generate_commentary(
            args.ollama_base_url,
            args.text_model,
            visual_description,
            vehicle_data,
        )
        record = {
            "index": index,
            "time_sec": time_sec,
            "frame": str(frame_path),
            "vehicle_data": vehicle_data,
            "visual_description": visual_description,
            "commentary": commentary,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "commentary.jsonl"
    md_path = output_dir / "commentary.md"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Sensor-grounded VLM commentary\n\n")
        f.write(f"- video: `{video}`\n")
        f.write(f"- vision_model: `{args.vision_model}`\n")
        f.write(f"- text_model: `{args.text_model}`\n")
        f.write(f"- crop: `{args.crop}`\n\n")
        for record in records:
            f.write(f"- {record['time_sec']:.1f}s: {record['commentary']}\n")


if __name__ == "__main__":
    main()
