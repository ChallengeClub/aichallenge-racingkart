#!/usr/bin/env python3
"""Benchmark local VLM tag extraction across camera preprocessing variants."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path
import time
import urllib.request

import cv2


def encode_jpeg_b64(image) -> str:
    ok, data = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        raise RuntimeError("failed to encode image")
    return base64.b64encode(data.tobytes()).decode("ascii")


def ollama_chat(base_url: str, payload: dict, timeout: int = 180) -> str:
    payload = {
        **payload,
        "stream": False,
        "options": {
            "temperature": payload.get("options", {}).get("temperature", 0.0),
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
        return value if isinstance(value, dict) else {"raw": text}
    except Exception:
        return {"raw": text}


def make_variants(camera):
    h, w = camera.shape[:2]
    lower = camera[h // 2 :, :]
    upper = camera[: h // 2, :]
    return {
        "lower_320x160_rgb": cv2.resize(lower, (320, 160), interpolation=cv2.INTER_AREA),
        "lower_160x80_rgb": cv2.resize(lower, (160, 80), interpolation=cv2.INTER_AREA),
        "upper_320x160_rgb": cv2.resize(upper, (320, 160), interpolation=cv2.INTER_AREA),
        "full_320x180_rgb": cv2.resize(camera, (320, 180), interpolation=cv2.INTER_AREA),
    }


def tag_image(base_url: str, model: str, image, variant: str) -> tuple[str, dict, float]:
    prompt = (
        "You are labeling a vehicle-mounted camera image for autonomous driving commentary. "
        "Return only compact JSON with these keys: "
        "road_direction, curve, barrier, sky, buildings, scene, confidence. "
        "Allowed road_direction values: left, right, straight, unknown. "
        "Allowed curve values: none, gentle, sharp, unknown. "
        "barrier, sky, buildings are booleans. "
        "scene is a short snake_case label. "
        "Do not mention UI, screenshots, RViz, simulator, or developers. "
        f"preprocess_variant={variant}"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [encode_jpeg_b64(image)],
            }
        ],
        "options": {"temperature": 0.0, "num_predict": 80},
    }
    started = time.perf_counter()
    text = ollama_chat(base_url, payload)
    elapsed = time.perf_counter() - started
    return text, parse_jsonish(text), elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--times-sec", default="8,16,24,32,40,48,56,64")
    parser.add_argument("--crop", default="168,664,452,300")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--vision-model", default="llava:7b")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    times = [float(item) for item in args.times_sec.split(",") if item.strip()]
    x, y, w, h = [int(value) for value in args.crop.split(",")]

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {args.video}")

    records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "vlm_preprocessing_tags.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    for time_sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000.0)
        ok, frame = cap.read()
        if not ok:
            print(f"skip failed frame time={time_sec}", flush=True)
            continue
        camera = frame[y : y + h, x : x + w]
        variants = make_variants(camera)
        for variant, image in variants.items():
            image_path = frame_dir / f"t{time_sec:06.2f}_{variant}.jpg"
            cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            raw, tags, latency = tag_image(args.ollama_base_url, args.vision_model, image, variant)
            record = {
                "time_sec": time_sec,
                "variant": variant,
                "image": str(image_path),
                "latency_sec": round(latency, 3),
                "tags": tags,
                "raw": raw,
            }
            records.append(record)
            with jsonl_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)

    (output_dir / "vlm_preprocessing_tags.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "vlm_preprocessing_tags.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "time_sec",
                "variant",
                "latency_sec",
                "road_direction",
                "curve",
                "barrier",
                "sky",
                "buildings",
                "scene",
                "confidence",
                "raw",
                "image",
            ],
        )
        writer.writeheader()
        for record in records:
            tags = record["tags"]
            writer.writerow(
                {
                    "time_sec": record["time_sec"],
                    "variant": record["variant"],
                    "latency_sec": record["latency_sec"],
                    "road_direction": tags.get("road_direction", ""),
                    "curve": tags.get("curve", ""),
                    "barrier": tags.get("barrier", ""),
                    "sky": tags.get("sky", ""),
                    "buildings": tags.get("buildings", ""),
                    "scene": tags.get("scene", ""),
                    "confidence": tags.get("confidence", ""),
                    "raw": record["raw"],
                    "image": record["image"],
                }
            )
    with (output_dir / "vlm_preprocessing_tags.md").open("w", encoding="utf-8") as file:
        file.write("# VLM preprocessing tag benchmark\n\n")
        file.write(f"- video: `{args.video}`\n")
        file.write(f"- crop: `{args.crop}`\n")
        file.write(f"- times_sec: `{args.times_sec}`\n")
        file.write(f"- vision_model: `{args.vision_model}`\n\n")
        for variant in sorted({record["variant"] for record in records}):
            rows = [record for record in records if record["variant"] == variant]
            latencies = [record["latency_sec"] for record in rows]
            if latencies:
                file.write(
                    f"## {variant}\n\n"
                    f"- latency_min: `{min(latencies):.3f}`\n"
                    f"- latency_median: `{sorted(latencies)[len(latencies)//2]:.3f}`\n"
                    f"- latency_max: `{max(latencies):.3f}`\n\n"
                )
            for record in rows:
                file.write(
                    f"- t={record['time_sec']:.2f}s latency={record['latency_sec']:.3f}s "
                    f"tags=`{json.dumps(record['tags'], ensure_ascii=False)}`\n"
                )
            file.write("\n")


if __name__ == "__main__":
    main()
