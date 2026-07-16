#!/usr/bin/env python3
"""Compare camera preprocessing variants with a local VLM commentary pipeline."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def ollama_chat(base_url: str, payload: dict) -> str:
    payload = {
        **payload,
        "stream": False,
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


def clean_sentence(text: str) -> str:
    text = " ".join(line.strip() for line in text.splitlines() if line.strip())
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    text = text.strip("「」\"' ")
    for sep in ("。", "！", "!"):
        if sep in text:
            return text.split(sep, 1)[0].strip() + "。"
    return text[:80]


def describe(base_url: str, model: str, image_path: Path, variant: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Describe the road scene in this vehicle-mounted camera preprocessing image. "
                    f"The preprocessing variant is {variant}. "
                    "Do not mention UI, screenshots, RViz, simulators, or developers. "
                    "Use one short English sentence."
                ),
                "images": [encode_image(image_path)],
            }
        ],
        "options": {"temperature": 0.0, "num_predict": 80},
    }
    return ollama_chat(base_url, payload)


def comment(base_url: str, model: str, description: str, variant: str, vehicle_data: dict) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only one concise Japanese commentary sentence. No thinking text.",
            },
            {
                "role": "user",
                "content": (
                    "あなたは自動運転カートの走行解説者です。"
                    "使える情報はvisual_descriptionとvehicle_dataだけです。"
                    "unknownの数値は言わないでください。45文字以内の日本語1文だけ。\n\n"
                    f"preprocessing_variant: {variant}\n"
                    f"visual_description: {description}\n"
                    f"vehicle_data: {json.dumps(vehicle_data, ensure_ascii=False)}"
                ),
            },
        ],
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 80},
    }
    return clean_sentence(ollama_chat(base_url, payload))


def make_variants(camera: np.ndarray) -> dict[str, np.ndarray]:
    h, w = camera.shape[:2]
    lower = camera[h // 2 :, :]

    lowres = cv2.resize(lower, (160, 80), interpolation=cv2.INTER_AREA)
    lowres_view = cv2.resize(lowres, (w, h // 2), interpolation=cv2.INTER_NEAREST)

    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    # Broad low-saturation road-like mask. This is intentionally simple.
    road_mask = cv2.inRange(hsv, (0, 0, 55), (179, 80, 230))
    road_bgr = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)

    return {
        "raw_camera": camera,
        "lower_half": lower,
        "lower_half_160x80_upscaled": lowres_view,
        "lower_half_edges": edges_bgr,
        "lower_half_road_mask": road_bgr,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-sec", type=float, default=15.0)
    parser.add_argument("--crop", default="168,664,452,300")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vision-model", default="llava:7b")
    parser.add_argument("--text-model", default="qwen3:8b")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_MSEC, args.time_sec * 1000.0)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"failed to read frame at {args.time_sec}s")

    x, y, w, h = [int(v) for v in args.crop.split(",")]
    camera = frame[y : y + h, x : x + w]
    variants = make_variants(camera)

    vehicle_data = {
        "time_sec": args.time_sec,
        "source": "cropped vehicle camera viewer subscribed to /sensing/camera/image_raw",
        "ego_speed_kmh": "unknown_in_this_recording",
        "v2x_distance_m": "unknown_in_this_recording",
        "controller": "pilot_net fixed mode in this source recording",
    }

    records = []
    for name, image in variants.items():
        image_path = frame_dir / f"{name}.jpg"
        cv2.imwrite(str(image_path), image)
        visual_description = describe(args.ollama_base_url, args.vision_model, image_path, name)
        commentary = comment(args.ollama_base_url, args.text_model, visual_description, name, vehicle_data)
        record = {
            "variant": name,
            "image": str(image_path),
            "visual_description": visual_description,
            "commentary": commentary,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

    (output_dir / "preprocess_compare.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "preprocess_compare.md").open("w", encoding="utf-8") as f:
        f.write("# Camera preprocessing VLM comparison\n\n")
        f.write(f"- time_sec: `{args.time_sec}`\n")
        f.write(f"- crop: `{args.crop}`\n")
        f.write(f"- vision_model: `{args.vision_model}`\n")
        f.write(f"- text_model: `{args.text_model}`\n\n")
        for record in records:
            f.write(f"## {record['variant']}\n\n")
            f.write(f"- visual: {record['visual_description']}\n")
            f.write(f"- commentary: {record['commentary']}\n")
            f.write(f"- image: `{record['image']}`\n\n")


if __name__ == "__main__":
    main()
