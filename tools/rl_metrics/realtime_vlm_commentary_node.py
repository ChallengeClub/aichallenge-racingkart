#!/usr/bin/env python3
"""Semi-realtime VLM commentary node for AI Challenge camera and vehicle topics."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
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


def ollama_chat(base_url: str, payload: dict, timeout: float = 180) -> str:
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


def ollama_generate(base_url: str, payload: dict, timeout: float = 180) -> str:
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


def visual_output_needs_fallback(text: str, visual_tags: dict) -> bool:
    cleaned = str(text or "").strip()
    if visual_tags:
        return False
    if not cleaned:
        return True
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return True
    if cleaned.count(",") >= 2 and all(ch in "0123456789.,-+[] eE" for ch in cleaned):
        return True
    return False


def derive_ambient_context(visual_tags: dict, visual_description: str) -> list[str]:
    tags: list[str] = []
    visual = str(visual_description or "").lower()

    explicit = visual_tags.get("ambient_context")
    if isinstance(explicit, list):
        tags.extend(str(item) for item in explicit if str(item).strip())
    elif isinstance(explicit, str) and explicit.strip():
        tags.extend(part.strip() for part in explicit.split(",") if part.strip())

    if visual_tags.get("barrier") or "barrier" in visual or "blue" in visual:
        tags.append("blue_white_barrier")
    if visual_tags.get("sky") or "sky" in visual or "cloud" in visual:
        tags.append("open_sky")
    if visual_tags.get("buildings") or "building" in visual or "city" in visual:
        tags.append("urban_buildings")
    scene = str(visual_tags.get("scene") or "").lower()
    if "sign" in scene or "sign" in visual or "billboard" in visual:
        tags.append("signboard")
    if "track" in scene or "narrow" in scene or "course" in visual:
        tags.append("narrow_sky_track")

    allowed = {
        "urban_buildings",
        "blue_white_barrier",
        "open_sky",
        "signboard",
        "narrow_sky_track",
    }
    deduped = []
    for tag in tags:
        normalized = tag.strip().lower()
        if normalized in allowed and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def parse_ambient_context(value: str) -> list[str]:
    allowed = {
        "urban_buildings",
        "blue_white_barrier",
        "open_sky",
        "signboard",
        "narrow_sky_track",
    }
    out = []
    for part in str(value or "").split(","):
        tag = part.strip().lower()
        if tag in allowed and tag not in out:
            out.append(tag)
    return out


VISIBLE_WORD_ALIASES = {
    "building": "building",
    "buildings": "building",
    "city": "city_buildings",
    "urban": "city_buildings",
    "stadium": "stadium",
    "roof": "roof",
    "crane": "crane",
    "cranes": "crane",
    "scaffold": "scaffolding",
    "scaffolding": "scaffolding",
    "billboard": "signboard",
    "sign": "signboard",
    "signboard": "signboard",
    "sky": "sky",
    "cloud": "sky",
    "barrier": "barrier",
    "barriers": "barrier",
    "blue-white barrier": "barrier",
    "blue and white barrier": "barrier",
    "blue_white_barrier": "barrier",
    "blue_and_white_barrier": "barrier",
    "track": "track",
    "road": "track",
    "course": "track",
}

LOW_INFORMATION_VISIBLE_WORDS = {"barrier", "track"}

SCENIC_WORD_PHRASES = {
    "building": ("建物を背景に", "奥の建物が見える中"),
    "city_buildings": ("建物を背景に", "コースの向こうに建物が見える中"),
    "stadium": ("スタンドを横に", "スタンドが見える区間で"),
    "signboard": ("看板が見える区間で", "コース脇の看板を見ながら"),
    "sky": ("開けた空の下",),
    "roof": ("屋根が見える区間で",),
    "crane": ("クレーンが見える区間で",),
    "scaffolding": ("建物を背景に",),
}


def normalize_visible_words(visual_tags: dict, visual_description: str) -> list[str]:
    words: list[str] = []
    raw = visual_tags.get("visible_words") if isinstance(visual_tags, dict) else None
    if isinstance(raw, list):
        words.extend(str(item) for item in raw if str(item).strip())
    elif isinstance(raw, str):
        words.extend(part.strip() for part in raw.split(",") if part.strip())

    visual = str(visual_description or "").lower()
    for phrase, normalized in VISIBLE_WORD_ALIASES.items():
        if phrase in visual:
            words.append(normalized)

    deduped = []
    for word in words:
        normalized = str(word).strip().lower().replace(" ", "_").replace("-", "_")
        normalized = VISIBLE_WORD_ALIASES.get(normalized, normalized)
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped[:8]


def visual_words_summary(visual_tags: dict, visual_description: str) -> dict:
    visible_words = normalize_visible_words(visual_tags, visual_description)
    salient_words = [word for word in visible_words if word not in LOW_INFORMATION_VISIBLE_WORDS]
    return {
        "visible_words": visible_words,
        "salient_words": salient_words,
        "low_information_words": [word for word in visible_words if word in LOW_INFORMATION_VISIBLE_WORDS],
        "scene_position": visual_tags.get("scene_position") if isinstance(visual_tags, dict) else None,
        "confidence": visual_tags.get("confidence") if isinstance(visual_tags, dict) else None,
        "driving_focus": visual_tags.get("driving_focus") if isinstance(visual_tags, dict) else None,
    }


def load_course_context(path: str) -> dict:
    if not path:
        return {"features": []}
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return {"features": payload}
    if isinstance(payload, dict):
        features = payload.get("features", [])
        if not isinstance(features, list):
            raise ValueError("course context JSON must contain a features list")
        return payload
    raise ValueError("course context JSON must be an object or a feature list")


def lookup_course_context(course_context: dict, vehicle_data: dict) -> dict:
    features = course_context.get("features", []) if isinstance(course_context, dict) else []
    ego = vehicle_data.get("ego", {})
    pose_x = ego.get("pose_x")
    pose_y = ego.get("pose_y")
    if pose_x is None or pose_y is None or not features:
        return {}

    matches = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        fx = feature.get("x", feature.get("pose_x"))
        fy = feature.get("y", feature.get("pose_y"))
        if fx is None or fy is None:
            continue
        try:
            distance = math.hypot(float(pose_x) - float(fx), float(pose_y) - float(fy))
            radius = float(feature.get("radius_m", course_context.get("default_radius_m", 12.0)))
        except (TypeError, ValueError):
            continue
        if distance <= radius:
            matches.append((distance, feature))

    if not matches:
        return {}
    distance, feature = sorted(matches, key=lambda item: item[0])[0]
    keep_keys = (
        "id",
        "name",
        "section",
        "corner_type",
        "tags",
        "risk_level",
        "ideal_line",
        "commentary_hints",
        "notes",
        "driving_strategy",
        "risk",
        "visual_landmarks",
    )
    result = {key: feature[key] for key in keep_keys if key in feature}
    result["distance_m"] = round(distance, 2)
    return result


def infer_driving_phase(vehicle_data: dict, history: list[dict]) -> str:
    ego = vehicle_data.get("ego", {})
    control = vehicle_data.get("control_cmd", {})
    accel = vehicle_data.get("acceleration", {})
    speed = float(ego.get("speed_kmh", 0.0) or 0.0)
    speed_error = float(vehicle_data.get("speed_tracking_error_kmh", 0.0) or 0.0)
    steering = float(control.get("steering_tire_angle_rad", 0.0) or 0.0)
    steering_abs = abs(steering)
    actual_accel = float(accel.get("actual_accel_mps2", 0.0) or 0.0)
    has_moved = bool(vehicle_data.get("vehicle_has_moved"))

    previous_steering_abs = steering_abs
    previous_speed = speed
    if history:
        previous = history[-1]
        previous_control = previous.get("control_cmd", {})
        previous_ego = previous.get("ego", {})
        previous_steering_abs = abs(float(previous_control.get("steering_tire_angle_rad", steering) or 0.0))
        previous_speed = float(previous_ego.get("speed_kmh", speed) or 0.0)
        has_moved = has_moved or any(
            float(row.get("ego", {}).get("speed_kmh", 0.0) or 0.0) > 5.0
            for row in history[-80:]
        )

    if speed < 1.0:
        return "stuck" if has_moved or speed_error > 8.0 else "recovery"
    if speed_error > 12.0 and speed < 5.0 and actual_accel < 0.05:
        return "stuck"
    if steering_abs <= 0.12:
        return "straight"
    if steering_abs > 0.18 and actual_accel < -0.15:
        return "entry"
    if steering_abs > 0.18 and steering_abs >= previous_steering_abs - 0.03:
        return "mid_corner"
    if steering_abs > 0.12 and (steering_abs < previous_steering_abs - 0.03 or speed > previous_speed + 0.8):
        return "exit"
    return "recovery"


def build_driving_state(vehicle_data: dict, vehicle_history: list[dict], course_context: dict) -> dict:
    ego = vehicle_data.get("ego", {})
    control = vehicle_data.get("control_cmd", {})
    accel = vehicle_data.get("acceleration", {})
    speed = float(ego.get("speed_kmh", 0.0) or 0.0)
    steering = float(control.get("steering_tire_angle_rad", 0.0) or 0.0)
    actual_accel = float(accel.get("actual_accel_mps2", 0.0) or 0.0)
    yaw_accel = float(accel.get("actual_yaw_accel_mps2", 0.0) or 0.0)
    speed_delta = 0.0
    if vehicle_history:
        previous_speed = float(vehicle_history[-1].get("ego", {}).get("speed_kmh", speed) or 0.0)
        speed_delta = speed - previous_speed
    return {
        "phase": infer_driving_phase(vehicle_data, vehicle_history),
        "speed_delta_kmh": round(speed_delta, 2),
        "steer_abs_rad": round(abs(steering), 3),
        "steer_direction": "left" if steering > 0.12 else "right" if steering < -0.12 else "straight",
        "accel_mps2": round(actual_accel, 3),
        "yaw_accel_abs": round(abs(yaw_accel), 3),
        "section": course_context.get("id") or course_context.get("section"),
        "course_tags": course_context_tags(course_context),
    }


def course_focuses_from_notes(notes: str) -> list[str]:
    text = str(notes or "")
    focuses = []
    if "アウト" in text and ("イン・アウト" in text or "インアウト" in text or "イン・イン" in text):
        focuses.append("out_in_out")
    if "しっかり減速" in text or "減速" in text:
        focuses.append("brake_entry")
    if "膨ら" in text:
        focuses.append("hold_inside_exit")
    if "ブレーキ不要" in text:
        focuses.append("no_brake_flow")
    if "アクセル全開" in text or "全開" in text:
        focuses.append("full_throttle")
    if "加速" in text:
        focuses.append("accel_zone")
    return focuses


def course_context_tags(course_context: dict) -> list[str]:
    tags = list(course_context.get("tags") or [])
    strategy = course_context.get("driving_strategy") or {}
    risk = course_context.get("risk") or {}
    for value in (
        strategy.get("priority"),
        strategy.get("line"),
        strategy.get("brake_timing"),
        *(risk.get("failure_modes") or []),
        *(course_context.get("visual_landmarks") or []),
    ):
        if value and value not in tags:
            tags.append(value)
    return tags


def course_note_phrases(notes: str) -> list[str]:
    focuses = set(course_focuses_from_notes(notes))
    candidates = []
    if "out_in_out" in focuses:
        candidates.append("アウトインアウトで、出口を広く使いたい区間だね。")
    if "brake_entry" in focuses:
        candidates.append("入口でしっかり減速して、向きを作りたい区間だね。")
    if "hold_inside_exit" in focuses:
        candidates.append("出口で外へ膨らみすぎないか、ここは見どころだね。")
    if "no_brake_flow" in focuses:
        candidates.append("ここはブレーキを我慢して、リズムを保ちたいね。")
    if "full_throttle" in focuses:
        candidates.append("慣れてきたらアクセル全開でつなげたい区間だね。")
    if "accel_zone" in focuses:
        candidates.append("ここから速度を乗せていく区間だね。")
    return candidates


def course_context_phrases(course_context: dict, driving_state: dict, commentary_type: str) -> list[str]:
    if not course_context:
        return []
    tags = set(course_context_tags(course_context))
    phase = str(driving_state.get("phase") or "")
    strategy = course_context.get("driving_strategy") or {}
    risk = course_context.get("risk") or {}
    ideal_line = str(course_context.get("ideal_line") or strategy.get("line") or "")
    risk_level = course_context.get("risk_level") or risk.get("level")
    visit_count = int(driving_state.get("section_visit_count", 1) or 1)
    speed_delta = driving_state.get("section_entry_speed_delta_kmh")
    focuses = set(course_focuses_from_notes(course_context.get("notes", "")))
    candidates = []

    for hint in course_context.get("commentary_hints") or []:
        if hint and hint not in candidates:
            candidates.append(str(hint))

    if visit_count > 1:
        if isinstance(speed_delta, (int, float)) and speed_delta > 1.0:
            candidates.append("さっきより入口の速度は乗ってる。出口の収まりに注目だね。")
        elif isinstance(speed_delta, (int, float)) and speed_delta < -1.0:
            candidates.append("今回は少し抑えて入ったね。出口を小さくまとめたい。")
        else:
            candidates.append("同じ区間の2回目、さっきとの差が見どころだね。")
    else:
        if "out_in_out" in focuses:
            candidates.append("ここは出口を広く使えるかが見どころだね。")
        if "brake_entry" in focuses:
            candidates.append("入口で速度を落として、先に向きを作りたい区間だね。")
        if "hold_inside_exit" in focuses:
            candidates.append("出口で外へ流れすぎないか、ここは見ておきたいね。")
        if "no_brake_flow" in focuses:
            candidates.append("ここはリズムを保って、ブレーキを我慢したい区間だね。")
        if "accel_zone" in focuses or "full_throttle" in focuses:
            candidates.append("ここから前へ伸ばせるか、加速のつなぎが大事だね。")

    if "crash_hotspot" in tags or risk_level in ("high", "critical"):
        candidates.append("ここはミスが出やすいカーブ、早めの姿勢作りだね。")
    if "out_in_out" in tags or "out-in-out" in ideal_line.lower() or "アウトインアウト" in ideal_line:
        candidates.append("アウトインアウトで、出口を広く使いたい区間だね。")
    if "exit_speed" in tags:
        candidates.append("ここは出口の伸びを残したい区間だね。")
    if "overtake_sensitive" in tags:
        candidates.append("ここは無理に詰めず、出口の伸びを残したいね。")
    if driving_state.get("section_entered") and candidates:
        return candidates
    if phase == "entry":
        candidates.append("入口は姿勢作り優先、ここで焦らないのが大事だね。")
    elif phase == "mid_corner":
        candidates.append("カーブ中盤、舵を急がせないのが見どころだね。")
    elif phase == "exit":
        candidates.append("出口でどこまで速度を乗せるか、ここが勝負だね。")
    return candidates


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
        "減速、加速",
        "優先して話す",
        "発話方針",
        "禁止語",
        "良い例",
        "悪い例",
        "自然に取り入れ",
        "車速",
        "ではなく",
        "こと。",
        "景色:",
        "景色：",
        "景色が",
        "助手席ai",
        "最近のコース",
        "聞こえます",
        "安全に",
        "走行中です",
        "コースを抜けながら",
        "ビルの間から",
        "直線とカーブを繰り返しています",
        "目標速度へ向けて加速しています",
        "道路の一部",
        "急激な加速",
        "速度を上昇",
        "背景に見える直線",
        "建物が背景に見える",
        "ビルが背景に見える",
        "レースは",
        "ビルを通過",
        "前方のビル",
        "建物の間を通り抜け",
        "建物の間を通過",
        "建物の間を通",
        "建物から抜け",
        "建物を抜け",
        "建物を通過",
        "から抜けてきた",
        "通り抜ける",
        "通っていき",
        "建物から、",
        "背景の建物から",
        "直線の後ろ",
        "前方を抜け",
        "出口で出口で",
        "重きを置",
        "重点を置",
        "重視",
        "抜け出すタイミング",
        "後ろに横目",
        "に横目に",
        "横目",
        "自動運転カート",
        "自動運転カートは",
        "自動運転カートが",
        "カートは",
        "発進直後",
        "スタート直後",
        "流れを切らさず",
        "流れを切らさない",
        "作りる",
        "備えりる",
        "保ちりる",
        "いきる",
        "進みる",
        "次の。",
        "へつ。",
        "区間へつ",
        "[",
        "]",
    )
    if not text.strip() or any(word in lowered for word in banned):
        return True
    if ("建物" in text or "ビル" in text) and any(word in text for word in ("通", "抜け", "出る", "出た")):
        return True
    return False


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


COMMENTARY_TOPIC_KEYWORDS = {
    "barrier": ("バリア", "ガードレール"),
    "steering": ("舵", "ライン", "向き", "コーナー", "カーブ", "旋回"),
    "deceleration": ("減速", "速度を落", "ブレーキ"),
    "acceleration": ("加速", "速度を乗", "伸び", "前へ"),
    "exit": ("出口", "立ち上がり"),
    "landmark": ("建物", "ビル", "看板", "クレーン", "空", "景色", "街並み", "コースの向こう"),
    "posture": ("姿勢", "安定", "落ち着"),
    "rhythm": ("リズム", "つな", "次の区間"),
}


COMMENTARY_PHRASE_KEYWORDS = {
    "landmark_visible": (
        "建物が見える中",
        "ビルが見える中",
        "建物を背景",
        "ビルを背景",
        "コースの向こうに建物",
        "コースの向こうにビル",
        "コースの向こうにクレーン",
    ),
    "line_keep": (
        "ラインを保",
        "ラインを作",
        "ラインを落ち着",
        "ライン優先",
    ),
    "posture_make": (
        "姿勢を作",
        "姿勢を整",
        "姿勢を崩さ",
        "姿勢が整",
    ),
    "next_connect": (
        "次へつな",
        "次の区間へつな",
        "前へつな",
    ),
    "flow_forward": (
        "流れを切らさず",
        "流れを切らさない",
        "前へ進んでる",
        "前へ進みます",
        "前へ進む",
    ),
    "exit_prepare": (
        "出口へ向けて",
        "出口で速度",
        "出口の伸び",
    ),
    "deceleration_phrase": (
        "減速を入れ",
        "速度を落",
        "ブレーキ",
    ),
    "pace_keep": (
        "ここは落ち着いて",
        "ペースを保",
        "落ち着いて、ペース",
    ),
    "align_next": (
        "向きを整",
        "次を合わせ",
        "車両の動きを見",
    ),
}


def commentary_topics(text: str) -> list[str]:
    topics = []
    for topic, keywords in COMMENTARY_TOPIC_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            topics.append(topic)
    return topics


def commentary_phrase_categories(text: str) -> list[str]:
    categories = []
    for category, keywords in COMMENTARY_PHRASE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            categories.append(category)
    return categories


VISUAL_WORD_REQUIREMENTS = {
    "建物": ("building", "city_buildings"),
    "ビル": ("building", "city_buildings"),
    "クレーン": ("crane",),
    "看板": ("signboard",),
    "スタジアム": ("stadium",),
    "空": ("sky", "open_sky"),
}


def commentary_uses_unavailable_visual_word(text: str, visual_summary: dict | None) -> bool:
    if not visual_summary:
        return False
    salient_words = set(str(word) for word in (visual_summary.get("salient_words") or []))
    for word, allowed_salient in VISUAL_WORD_REQUIREMENTS.items():
        if word in text and not any(item in salient_words for item in allowed_salient):
            return True
    return False


def commentary_is_low_information(text: str) -> bool:
    stripped = text.strip()
    normalized = commentary_repeat_key(stripped)
    low_information_exact = {
        "走行フェーズは直線だね",
        "今の直線で走ってるね",
        "今の直線で走っている",
        "直線で走ってるね",
        "直線で走っている",
        "建物が見えますね",
        "建物が見えるね",
        "ビルが見えますね",
        "ビルが見えるね",
        "クレーンが見えますね",
        "クレーンが見えるね",
    }
    if normalized in low_information_exact:
        return True
    if len(normalized) <= 14 and any(word in stripped for word in ("建物", "ビル", "クレーン", "看板", "空")):
        return True
    if "直線で走って" in stripped and any(word in stripped for word in ("建物", "ビル", "クレーン", "看板", "空", "景色")):
        return True
    if "背景に見える" in stripped and any(word in stripped for word in ("建物", "ビル", "クレーン", "看板")):
        return True
    if "ここは落ち着いて" in stripped and "ペースを保" in stripped:
        return True
    if "走行フェーズ" in stripped:
        return True
    if re.fullmatch(r"(今の)?(直線|コーナー|カーブ)で走って(る|いる)ね?", normalized):
        return True
    return False


def recent_topic_summary(recent: list[str]) -> dict:
    recent_rows = []
    counts = {topic: 0 for topic in COMMENTARY_TOPIC_KEYWORDS}
    for text in recent[-6:]:
        topics = commentary_topics(text)
        recent_rows.append({"text": text, "topics": topics})
        for topic in topics:
            counts[topic] = counts.get(topic, 0) + 1
    avoid_topics = []
    for text in recent[-2:]:
        avoid_topics.extend(commentary_topics(text))
    avoid_topics = list(dict.fromkeys(avoid_topics))
    return {
        "recent": recent_rows,
        "counts": {topic: count for topic, count in counts.items() if count},
        "avoid_topics": avoid_topics,
    }


def commentary_repeats_recent_topic(text: str, recent: list[str]) -> bool:
    topics = set(commentary_topics(text))
    if not topics:
        return False
    recent_topics = set()
    for item in recent[-2:]:
        recent_topics.update(commentary_topics(item))
    return bool(topics & recent_topics)


def commentary_repeats_recent_phrase(text: str, recent: list[str]) -> bool:
    categories = set(commentary_phrase_categories(text))
    if not categories:
        return False
    recent_categories = set()
    for item in recent[-8:]:
        recent_categories.update(commentary_phrase_categories(item))
    return bool(categories & recent_categories)


def ambient_phrase(ambient_context: list[str]) -> str:
    if "urban_buildings" in ambient_context:
        return "コース周辺の景色を見ながら"
    if "open_sky" in ambient_context:
        return "開けた空の下"
    if "signboard" in ambient_context:
        return "正面の看板を見ながら"
    if "narrow_sky_track" in ambient_context:
        return "SKY TRACKらしい短い区間"
    return ""


def maybe_add_ambient_prefix(text: str, ambient_context: list[str], index: int, commentary_type: str) -> str:
    if not ambient_context:
        return text
    if commentary_type == "ambient":
        return text
    prefix = ambient_phrase(ambient_context)
    if not prefix or text.startswith(prefix):
        return text
    trimmed = text.rstrip("。")
    if len(prefix) + len(trimmed) + 2 > 42:
        return text
    return f"{prefix}、{trimmed}。"


def scenic_phrase_candidates(visual_summary: dict | None, recent: list[str]) -> list[str]:
    if not visual_summary:
        return []
    salient_words = [
        str(word)
        for word in (visual_summary.get("salient_words") or [])
        if str(word) in SCENIC_WORD_PHRASES
    ]
    if not salient_words:
        return []

    recent_text = " ".join(recent[-3:])
    out = []
    for word in salient_words:
        for phrase in SCENIC_WORD_PHRASES.get(word, ()):
            if phrase and phrase not in recent_text and phrase not in out:
                out.append(phrase)
    return out


def maybe_blend_scene_phrase(
    text: str,
    visual_summary: dict | None,
    recent: list[str],
    max_chars: int,
    commentary_type: str,
    index: int,
) -> str:
    if commentary_type in ("recovery", "critical"):
        return text
    if commentary_is_low_information(text):
        return text
    if commentary_repeats_recent_topic(text, recent) and not scenic_phrase_candidates(visual_summary, recent):
        return text

    candidates = scenic_phrase_candidates(visual_summary, recent)
    if not candidates:
        return text
    recent_topics = set()
    for item in recent[-3:]:
        recent_topics.update(commentary_topics(item))
    should_blend = (
        commentary_type == "ambient"
        or "landmark" not in recent_topics
        or index % 3 == 0
    )
    if not should_blend:
        return text

    base = clean_commentary(text, max_chars, 1).rstrip("。")
    if any(word in base for word in ("建物", "ビル", "クレーン", "看板", "スタンド", "空", "屋根")):
        return text
    for phrase in candidates:
        candidate = clean_commentary(f"{phrase}、{base}。", max_chars, 1)
        if (
            not commentary_needs_retry(candidate)
            and not commentary_uses_unavailable_visual_word(candidate, visual_summary)
            and not commentary_is_low_information(candidate)
            and not commentary_repeats_any(candidate, recent)
            and not commentary_repeats_recent_phrase(candidate, recent)
        ):
            return candidate
    return text


def casualize_tameguchi(text: str) -> str:
    cleaned = text
    replacements = [
        ("少し前の", "今の"),
        ("少し前、", ""),
        ("今の減速は、", "今の減速、"),
        ("今の加速は、", "今の加速、"),
        ("今の舵は、", "今の舵、"),
        ("作ります。", "作ってる。"),
        ("備えます。", "備えてる。"),
        ("保ちます。", "保ってる。"),
        ("乗せます。", "乗せてる。"),
        ("見ます。", "見てる。"),
        ("いきます。", "いく。"),
        ("行きます。", "行く。"),
        ("詰まります。", "詰まるね。"),
        ("鈍いです。", "鈍いね。"),
        ("しています。", "してる。"),
        ("ています。", "てる。"),
        ("います。", "いる。"),
        ("でした。", "だった。"),
        ("です。", "だね。"),
        ("ます。", "る。"),
        ("備える。", "備えてる。"),
        ("整える。", "整えてる。"),
        ("つなぎる。", "つないでる。"),
        ("保ちる。", "保ってる。"),
        ("乗せる。", "乗せてる。"),
        ("見る。", "見てる。"),
        ("作る。", "作ってる。"),
        ("進みる。", "進んでる。"),
        ("重視された。", "見てたね。"),
        ("出口で出口で", "出口で"),
    ]
    for before, after in replacements:
        cleaned = cleaned.replace(before, after)
    return cleaned


PERSONA_PROFILES = {
    "passenger_casual_light": {
        "prompt": "人格は、横で一緒に走りを見ている助手席の相棒です。敬語ではなく、カジュアルなタメ口で短く話してください。不完全さの表現は控えめにしてください。",
        "recovery_prefix": "",
        "soften_recovery": True,
        "tameguchi": True,
    },
    "loose_mascot": {
        "prompt": "人格は、助手席にいる少し抜けたゆるキャラです。完璧な解説者ではなく、走りを見て素直に反応します。敬語は禁止です。短い感想は自然に混ぜてよいですが、同じ言い回しは繰り返さないでください。少し間違っても違和感がない、ゆるい相棒として話してください。",
        "recovery_prefix": "たぶん、",
        "soften_recovery": True,
        "tameguchi": True,
        "mascot": True,
    },
    "commentator_neutral": {
        "prompt": "人格は、視聴者向けの落ち着いたレース解説者です。親しみよりも分かりやすさを優先してください。",
        "recovery_prefix": "",
        "soften_recovery": False,
        "tameguchi": False,
    },
    "developer_technical": {
        "prompt": "人格は、自動運転の挙動を横で確認する開発者アシスタントです。操作理由と観察を短く述べてください。",
        "recovery_prefix": "見たところ、",
        "soften_recovery": True,
        "tameguchi": False,
    },
    "off": {
        "prompt": "人格付けはせず、短い走行実況だけを返してください。",
        "recovery_prefix": "",
        "soften_recovery": False,
        "tameguchi": False,
    },
}


def persona_profile(style: str) -> dict:
    return PERSONA_PROFILES.get(style, PERSONA_PROFILES["passenger_casual_light"])


def apply_persona_layer(text: str, commentary_type: str, max_chars: int, persona_style: str) -> str:
    profile = persona_profile(persona_style)
    if persona_style == "off":
        return clean_commentary(text, max_chars, 1)
    cleaned = clean_commentary(text, max_chars, 1)
    reflective_prefixes = (
        "今の",
        "ここは",
        "青白いバリア沿い",
        "正面の建設中の建物",
        "建設中の建物",
        "正面の看板",
        "開けた空の下",
        "SKY TRACK",
    )
    if commentary_type in ("why", "watch_point") and not cleaned.startswith(reflective_prefixes):
        candidate = f"今の{cleaned}"
        if len(candidate) <= max_chars:
            cleaned = candidate
    replacements = {
        "右方向へ旋回を始めます": "右コーナーの流れを見ています",
        "左方向へ旋回を始めます": "左コーナーの流れを見ています",
        "旋回を始めます": "向きの変化を見ています",
        "減速します": "減速を入れています",
        "加速します": "速度を乗せています",
        "確認します": "見ています",
        "走行します": "進めています",
    }
    for before, after in replacements.items():
        cleaned = cleaned.replace(before, after)
    if profile.get("tameguchi"):
        cleaned = casualize_tameguchi(cleaned)
    if profile.get("mascot"):
        mascot_replacements = {
            "速度を乗せてる": "出口で伸びてる",
            "ここから伸びていく区間だね": "ここから伸びる区間だね",
            "出口の伸び重視だった": "出口の伸びを見てた",
            "前が開いた場面の伸び重視だった": "前が開いて伸ばせた",
            "ラインを作ってる": "ラインを作ってるね",
            "舵を保ち": "舵を保って",
            "右コーナー中盤": "右コーナー中盤",
            "左コーナー中盤": "左コーナー中盤",
            "姿勢を整えてる": "向きを整えてる",
            "次の向き変えに備えてる": "出口へ向けて備えてる",
        }
        for before, after in mascot_replacements.items():
            cleaned = cleaned.replace(before, after)
    recovery_prefix = str(profile.get("recovery_prefix") or "")
    if (
        commentary_type == "recovery"
        and profile.get("soften_recovery")
        and recovery_prefix
        and len(cleaned) + len(recovery_prefix) <= max_chars
        and not cleaned.startswith(recovery_prefix)
    ):
        cleaned = f"{recovery_prefix}{cleaned}"
    return clean_commentary(cleaned, max_chars, 1)


def race_commentary_fallback(visual_description: str, vehicle_data: dict, recent: list[str]) -> str:
    ego = vehicle_data.get("ego", {})
    control = vehicle_data.get("control_cmd", {})
    accel = vehicle_data.get("acceleration", {})
    speed_error = float(vehicle_data.get("speed_tracking_error_kmh", 0.0) or 0.0)
    actual_accel = float(accel.get("actual_accel_mps2", 0.0) or 0.0)
    steering = float(control.get("steering_tire_angle_rad", 0.0) or 0.0)
    speed = float(ego.get("speed_kmh", 0.0) or 0.0)
    visual = visual_description.lower()

    candidates = []
    if speed < 1.0:
        candidates.extend([
            "ここは姿勢を落ち着かせています。",
            "車両の動きを見て、次を合わせています。",
        ])
    if actual_accel < -0.3:
        candidates.extend([
            "減速を入れて、出口へ向けて備えています。",
            "速度を落とし、姿勢を整えています。",
        ])
    if abs(steering) > 0.25:
        turn = "左" if steering > 0 else "右"
        candidates.extend([
            f"{turn}コーナー中盤、ラインを作っています。",
            f"{turn}への舵を保ち、姿勢を整えています。",
        ])
    if speed_error > 8.0 and actual_accel > 0.05:
        candidates.extend([
            "前方クリア、出口で速度を乗せます。",
            "今の加速いいね。",
            "姿勢が整い、ここから伸びていきます。",
        ])
    candidates.extend([
        "出口へ向けて、向きを整えます。",
        "立ち上がり重視で、ラインを保っています。",
        "ここは向きを整えて、次へつなぎます。",
        "姿勢が整い、次の区間へつなぎます。",
    ])
    if "barrier" in visual or "blue" in visual:
        candidates.extend([
            "コース脇を見ながら、舵を落ち着かせています。",
            "ラインを保って、次の区間へつなぎます。",
        ])
    for candidate in candidates:
        if (
            not commentary_needs_retry(candidate)
            and not commentary_repeats_any(candidate, recent)
            and not commentary_repeats_recent_topic(candidate, recent)
            and not commentary_repeats_recent_phrase(candidate, recent)
        ):
            return candidate
    return "車両の動きを見て、次を合わせています。"


def final_non_repeating_commentary(
    commentary: str,
    visual_description: str,
    visual_summary: dict,
    vehicle_data: dict,
    recent: list[str],
    max_chars: int,
    persona_style: str,
    commentary_type: str,
) -> str:
    candidate = clean_commentary(commentary, max_chars, 1)
    if (
        not commentary_needs_retry(candidate)
        and not commentary_uses_unavailable_visual_word(candidate, visual_summary)
        and not commentary_is_low_information(candidate)
        and not commentary_repeats_any(candidate, recent)
        and not commentary_repeats_recent_topic(candidate, recent)
        and not commentary_repeats_recent_phrase(candidate, recent)
    ):
        return candidate

    fallback = race_commentary_fallback(visual_description, vehicle_data, recent)
    fallback = apply_persona_layer(fallback, commentary_type, max_chars, persona_style)
    fallback = clean_commentary(fallback, max_chars, 1)
    if (
        not commentary_needs_retry(fallback)
        and not commentary_uses_unavailable_visual_word(fallback, visual_summary)
        and not commentary_is_low_information(fallback)
        and not commentary_repeats_any(fallback, recent)
        and not commentary_repeats_recent_topic(fallback, recent)
        and not commentary_repeats_recent_phrase(fallback, recent)
    ):
        return fallback

    control = vehicle_data.get("control_cmd", {})
    accel = vehicle_data.get("acceleration", {})
    speed_error = float(vehicle_data.get("speed_tracking_error_kmh", 0.0) or 0.0)
    actual_accel = float(accel.get("actual_accel_mps2", 0.0) or 0.0)
    steering = float(control.get("steering_tire_angle_rad", 0.0) or 0.0)
    turn = "左" if steering > 0 else "右"
    alternates = []
    if abs(steering) > 0.25:
        alternates.extend([
            f"{turn}の舵を保って、出口へつないでる。",
            f"{turn}コーナー、ラインを落ち着かせてる。",
            f"{turn}へ向きを整えて、次へつないでる。",
        ])
    if speed_error > 8.0 and actual_accel > 0.05:
        alternates.extend([
            "出口で速度を乗せて、前へ伸ばしてる。",
            "前が開いて、ここは伸びる場面だね。",
        ])
    if actual_accel < -0.3:
        alternates.extend([
            "減速を入れて、向き変えに備えてる。",
            "速度を落として、向きを整えてる。",
        ])
    alternates.extend([
        "ラインを保って、次の区間へつないでる。",
        "ここは姿勢を崩さず、前へつないでる。",
        "短い区間だけど、リズムは保ててる。",
        "バリア沿いに、舵を落ち着かせてる。",
    ])
    for alternate in alternates:
        alternate = clean_commentary(alternate, max_chars, 1)
        if (
            not commentary_needs_retry(alternate)
            and not commentary_uses_unavailable_visual_word(alternate, visual_summary)
            and not commentary_is_low_information(alternate)
            and not commentary_repeats_any(alternate, recent)
            and not commentary_repeats_recent_topic(alternate, recent)
            and not commentary_repeats_recent_phrase(alternate, recent)
        ):
            return alternate
    generic_alternates = [
        "車両の動きを見て、次を合わせてる。",
        "今は舵の戻りを見てる。",
        "短い直線、次のカーブへ備えてる。",
        "加速の入り方を見てる。",
    ]
    for alternate in generic_alternates:
        alternate = clean_commentary(alternate, max_chars, 1)
        if (
            not commentary_needs_retry(alternate)
            and not commentary_uses_unavailable_visual_word(alternate, visual_summary)
            and not commentary_is_low_information(alternate)
            and not commentary_repeats_any(alternate, recent)
            and not commentary_repeats_recent_topic(alternate, recent)
            and not commentary_repeats_recent_phrase(alternate, recent)
        ):
            return alternate
    return clean_commentary(candidate, max_chars, 1)


def tagged_race_commentary(
    visual_tags: dict,
    ambient_context: list[str],
    vehicle_data: dict,
    commentary_type: str,
    recent: list[str],
) -> str:
    ego = vehicle_data.get("ego", {})
    control = vehicle_data.get("control_cmd", {})
    accel = vehicle_data.get("acceleration", {})
    speed = float(ego.get("speed_kmh", 0.0) or 0.0)
    speed_error = float(vehicle_data.get("speed_tracking_error_kmh", 0.0) or 0.0)
    actual_accel = float(accel.get("actual_accel_mps2", 0.0) or 0.0)
    steering = float(control.get("steering_tire_angle_rad", 0.0) or 0.0)
    road_direction = str(visual_tags.get("road_direction") or "").lower()
    course_context = vehicle_data.get("course_context", {})
    driving_state = vehicle_data.get("driving_state", {})

    if steering > 0.18 or road_direction == "left":
        turn = "左"
    elif steering < -0.18 or road_direction == "right":
        turn = "右"
    else:
        turn = ""

    live_candidates = []
    why_candidates = []
    watch_candidates = []
    ambient_candidates = []
    recovery_candidates = [
        "出口へ向けて、向きを整えます。",
        "姿勢が整い、次の区間へつなぎます。",
        "ここはリズムよく、次の区間へつなぎます。",
    ]

    course_candidates = course_context_phrases(course_context, driving_state, commentary_type)
    if driving_state.get("phase") == "stuck":
        live_candidates[:0] = [
            "車速が落ちています。ここは止まった原因を見たいです。",
            "前へ出ていません。接触か姿勢の乱れを確認したいです。",
        ]
        watch_candidates.insert(0, "止まり方に注目です。ここは原因を切り分けたい場面です。")
    suppress_course_context = driving_state.get("phase") in {"stuck", "recovery"}
    if not suppress_course_context and driving_state.get("section_entered") and course_candidates:
        live_candidates[:0] = course_candidates[:1]
        watch_candidates[:0] = course_candidates[:1]
    if suppress_course_context:
        course_candidates = []
    if commentary_type == "watch_point":
        watch_candidates.extend(course_candidates)
    elif commentary_type == "why":
        why_candidates.extend(course_candidates)
    elif commentary_type == "sport_commentary":
        live_candidates.extend(course_candidates)
    elif commentary_type == "recovery":
        recovery_candidates.extend(course_candidates[:1])

    if commentary_type == "ambient" and ambient_context:
        if "blue_white_barrier" in ambient_context:
            ambient_candidates.extend([
                "コース脇を見ながら、姿勢を整えます。",
                "ラインを保って、次へつなぎます。",
            ])
        if "urban_buildings" in ambient_context:
            ambient_candidates.extend([
                "コースの向こうに景色が見えます。",
                "コース周辺の景色を見ながら、出口へ備えます。",
                "ここは向きを整えて、次へつなぎます。",
            ])
        if "open_sky" in ambient_context:
            ambient_candidates.append("空の下、短い直線へつなぎます。")
        if "signboard" in ambient_context:
            ambient_candidates.append("正面の看板を見ながら、ラインを保ちます。")
        if "narrow_sky_track" in ambient_context:
            ambient_candidates.append("屋外SKY TRACK、出口が大事です。")

    if speed < 1.0:
        live_candidates.extend([
            "スタート直後、前方のラインを見ています。",
            "発進直後、姿勢を落ち着かせています。",
        ])
        why_candidates.append("ここは無理せず、向きを整える発進でした。")
    if actual_accel < -0.3:
        live_candidates.extend([
            "減速を入れて、出口へ向けて備えています。",
            "速度を落とし、姿勢を整えています。",
        ])
        if turn:
            live_candidates.append(f"{turn}コーナー手前、早めに速度を落としています。")
        why_candidates.extend([
            "今の減速は、出口で向きを作るためでした。",
            "少し前の減速は、姿勢を崩さないためでした。",
        ])
        watch_candidates.append("ここは減速後の出口の伸びに注目です。")
    if abs(steering) > 0.25 and turn:
        live_candidates.extend([
            f"{turn}コーナー中盤、ラインを作っています。",
            f"{turn}への舵を保って、出口へつなぎます。",
            f"{turn}への舵を保ち、姿勢を整えています。",
        ])
        why_candidates.append(f"今の舵は、{turn}の出口を楽にする動きでした。")
        watch_candidates.append("ここは舵を戻すタイミングが見どころです。")
    if speed_error > 8.0 and actual_accel > 0.05:
        live_candidates.extend([
            "前方クリア、出口で速度を乗せています。",
            "今の加速いいね。",
            "姿勢が整い、ここから伸びていく区間です。",
        ])
        why_candidates.append("今の加速は、前が開いた場面の伸び重視でした。")
        watch_candidates.append("ここは出口でどこまで速度を乗せるかに注目です。")
    if commentary_type == "watch_point":
        watch_candidates.extend([
            "次の向き変え、ここは姿勢が大事です。",
            "出口で速度を乗せられるかに注目です。",
        ])
    if commentary_type == "live_call":
        live_candidates.extend([
            "ラインを保って、次の区間へ入っています。",
            "車両の向きが決まり、前へつないでいます。",
        ])
    if commentary_type == "why":
        why_candidates.extend([
            "ライン優先で、舵を急がせない判断です。",
            "次の出口を見て、姿勢を先に作ります。",
        ])
    if commentary_type == "recovery":
        recovery_candidates.extend([
            "状況を見ながら、次の動きへつないでいます。",
            "大きく乱さず、ペースを整えています。",
        ])
    if commentary_type == "sport_commentary":
        live_candidates.extend([
            "立ち上がり重視で、ラインを保っています。",
            "ここはリズムよく、次の区間へつなぎます。",
        ])

    by_type = {
        "live_call": live_candidates,
        "why": why_candidates,
        "watch_point": watch_candidates,
        "ambient": ambient_candidates,
        "recovery": recovery_candidates,
        "sport_commentary": live_candidates,
    }
    candidates = by_type.get(commentary_type, []) + live_candidates + why_candidates + watch_candidates + recovery_candidates

    for candidate in candidates:
        candidate = maybe_add_ambient_prefix(candidate, ambient_context, len(recent), commentary_type)
        if (
            not commentary_needs_retry(candidate)
            and not commentary_repeats_any(candidate, recent)
            and not commentary_repeats_recent_phrase(candidate, recent)
        ):
            return candidate
    return "ラインを保って、次へつなぎます。"


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
    course_context = vehicle_data.get("course_context", {})
    driving_state = vehicle_data.get("driving_state", {})
    course_phrases = course_context_phrases(course_context, driving_state, "live_call")

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
        if event_type == "impact_like":
            if course_context:
                return "今、接触して姿勢が乱れたね。"
            return "今、接触か大きな姿勢乱れ。"
        if event_type == "stuck":
            return "止まっています。復帰を見たい。"
        if event_type == "stuck_persistent":
            return "まだ動けていない。復帰が課題。"
        if course_phrases and event_type == "section_enter":
            return course_phrases[0]
        if event_type == "start":
            return "スタート直後、前方のラインを見ています。"
        if event_type == "restart":
            return "止まった後、もう一度立て直しています。"
        if event_type == "accel_start":
            return "前方が開けて、速度を乗せる区間です。"
        if event_type == "brake_start":
            if abs(steering) > 0.2 or "curve" in visual or "left" in visual or "right" in visual:
                return f"{direction}手前、速度を落としてラインを整えています。"
            return "減速を入れて、車両姿勢を整えています。"
        if event_type == "turn_start":
            turn = "左" if steering > 0 else "右"
            return f"{turn}コーナー中盤、ラインを作っています。"
        if event_type == "turn_end":
            return "舵が戻り、出口で速度を乗せています。"
        if event_type == "target_speed_reached":
            return "ペースが合ってきました。流れを保っています。"
        if event_type == "barrier_curve":
            return f"{direction}の流れです。バリア沿いにラインを合わせています。"
        if speed_error > 8.0 and actual_accel > 0.05:
            return "前方クリア、速度を乗せる場面です。"
        if actual_accel < -0.3:
            return "減速を入れて、次のラインへ整えています。"
        return "動きが変わりました。私は車両の向きを見ています。"

    if style == "scenery":
        barrier_visible = "barrier" in visual and "no visible barrier" not in visual and "no barriers" not in visual
        if barrier_visible:
            return "コース脇には青白いバリアが続いています。"
        if "crane" in visual:
            return "上空にはクレーンが見えます。イベント会場の中を走っている雰囲気です。"
        if "cityscape" in visual or "building" in visual or "structure" in visual:
            return "周囲には大きな建物や街並みが見えます。屋外コースの景色です。"
        if "tree" in visual or "grass" in visual or "greenery" in visual or "green" in visual:
            return "コース周辺に緑が見えます。青空の下を抜けていきます。"
        if "road" in visual or "track" in visual:
            return "前方にはコースが続いています。"
        if "sky" in visual or "cloud" in visual:
            return "空が広く見えます。開けたコースを走っています。"
        return "車載カメラの上側から、周囲の景色を確認しています。"

    if speed < 1.0:
        return "スタート直後、前方のコースを確認しています。"
    if course_phrases and (
        actual_accel < -0.2
        or abs(steering) > 0.18
        or speed_error > 6.0
        or "curve" in visual
        or "left" in visual
        or "right" in visual
    ):
        return course_phrases[0]
    if actual_accel < -0.3:
        if abs(steering) > 0.25 or "curve" in visual or "left" in visual or "right" in visual:
            return f"減速しながら{direction}に備えています。"
        return "速度を落として姿勢を整えています。"
    if abs(steering) > 0.3:
        turn = "左" if steering > 0 else "右"
        return f"{turn}へ舵を入れ、ラインを整えています。"
    if speed_error > 8.0 and actual_accel > 0.05:
        return "前方クリア、出口で速度を乗せます。"
    if actual_accel > 0.15:
        return f"{direction}へ向けて、姿勢を整えます。"
    if "barrier" in visual or "blue" in visual:
        return "バリア沿いに、舵を落ち着かせています。"
    return "姿勢が整い、次の区間へつなぎます。"


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
        self.initial_generation_done = False
        self.initial_timer = None
        self.last_event_data = None
        self.last_spoken_wall_time = 0.0
        self.last_critical_event_wall_time = 0.0
        self.vehicle_has_moved = False
        self.active_stuck_spoken = False
        self.stuck_session_start_wall_time = 0.0
        self.stuck_persistent_spoken = False
        self.stop_after_moving_start_wall_time = 0.0
        self.impact_suppress_stuck_until = 0.0
        self.speed_recovery_start_wall_time = 0.0
        self.last_generated_image_hash = ""
        self.last_commentary = ""
        self.last_visual_description = ""
        self.recent_commentaries = []
        self.mentioned_ambient_tags: set[str] = set()
        self.vehicle_history = []
        self.current_course_section = ""
        self.first_course_section = ""
        self.first_course_section_wall_time = 0.0
        self.course_lap_estimate = 1
        self.course_section_stats: dict[str, dict] = {}
        self.previous_vlm_image_b64 = ""
        self.previous_vlm_image = None
        self.course_context = load_course_context(args.course_context)

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
        if args.commentary_trigger == "interval":
            self.initial_timer = self.create_timer(0.5, self.on_initial_timer)
        self.get_logger().info(
            f"started interval={args.interval_sec}s preprocess={args.preprocess} "
            f"course_features={len(self.course_context.get('features', []))} output={self.output_dir}"
        )

    def ollama_timeout_sec(self, started: float) -> float:
        timeout = max(0.1, float(self.args.ollama_timeout_sec))
        if self.args.generation_deadline_sec <= 0:
            return timeout
        remaining = float(self.args.generation_deadline_sec) - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError(f"generation deadline exceeded before Ollama call: {self.args.generation_deadline_sec}s")
        return max(0.1, min(timeout, remaining))

    def check_generation_deadline(self, started: float, stage: str) -> None:
        if self.args.generation_deadline_sec <= 0:
            return
        elapsed = time.perf_counter() - started
        if elapsed > self.args.generation_deadline_sec:
            raise TimeoutError(
                f"generation deadline exceeded after {stage}: "
                f"{elapsed:.3f}s > {self.args.generation_deadline_sec:.3f}s"
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

    def vehicle_start_detected_locked(self) -> bool:
        if self.latest_odom is not None:
            speed_mps = float(self.latest_odom.twist.twist.linear.x)
            if speed_mps * 3.6 > 1.0:
                return True
        return False

    def try_start_generation(self, log_waiting: bool = False, require_vehicle_start: bool = False) -> bool:
        if self.busy:
            return False
        with self.lock:
            if self.latest_image is None:
                if log_waiting:
                    self.get_logger().info("waiting for image")
                return False
            if require_vehicle_start and not self.vehicle_start_detected_locked():
                if log_waiting:
                    self.get_logger().info("waiting for vehicle start")
                return False
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
        return True

    def on_initial_timer(self) -> None:
        if self.initial_generation_done:
            if self.initial_timer is not None:
                self.initial_timer.cancel()
            return
        if self.try_start_generation(log_waiting=False, require_vehicle_start=True):
            self.initial_generation_done = True
            if self.initial_timer is not None:
                self.initial_timer.cancel()

    def on_timer(self) -> None:
        require_vehicle_start = self.args.commentary_trigger == "interval" and not self.initial_generation_done
        if (
            self.try_start_generation(log_waiting=True, require_vehicle_start=require_vehicle_start)
            and not self.initial_generation_done
        ):
            self.initial_generation_done = True
            if self.initial_timer is not None:
                self.initial_timer.cancel()

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

    def annotate_course_progress(self, data: dict, driving_state: dict) -> dict:
        section = str(driving_state.get("section") or "")
        now = float(data.get("wall_time_sec", time.time()) or time.time())
        speed = float(data.get("ego", {}).get("speed_kmh", 0.0) or 0.0)
        driving_state.setdefault("lap_estimate", self.course_lap_estimate)
        driving_state.setdefault("section_visit_count", 0)
        driving_state.setdefault("section_entered", False)
        driving_state.setdefault("section_entry_speed_delta_kmh", None)

        if not section:
            self.current_course_section = ""
            return driving_state

        stats = self.course_section_stats.setdefault(
            section,
            {
                "visits": 0,
                "last_entry_speed_kmh": None,
                "last_enter_wall_time": 0.0,
            },
        )
        entered = section != self.current_course_section
        if entered:
            if not self.first_course_section:
                self.first_course_section = section
                self.first_course_section_wall_time = now
            elif (
                section == self.first_course_section
                and int(stats.get("visits", 0) or 0) > 0
                and now - self.first_course_section_wall_time > 45.0
            ):
                self.course_lap_estimate += 1
                self.first_course_section_wall_time = now

            last_entry_speed = stats.get("last_entry_speed_kmh")
            speed_delta = None if last_entry_speed is None else round(speed - float(last_entry_speed), 2)
            stats["visits"] = int(stats.get("visits", 0) or 0) + 1
            stats["last_entry_speed_kmh"] = speed
            stats["last_enter_wall_time"] = now
            self.current_course_section = section

            driving_state["section_entered"] = True
            driving_state["section_entry_speed_delta_kmh"] = speed_delta

        driving_state["section_visit_count"] = int(stats.get("visits", 0) or 0)
        driving_state["lap_estimate"] = self.course_lap_estimate
        return driving_state

    def detect_event(self, data: dict) -> tuple[str, str]:
        now = float(data.get("wall_time_sec", time.time()))
        ego = data.get("ego", {})
        control = data.get("control_cmd", {})
        accel = data.get("acceleration", {})
        driving_state = data.get("driving_state", {})
        speed = float(ego.get("speed_kmh", 0.0) or 0.0)
        speed_error = float(data.get("speed_tracking_error_kmh", 0.0) or 0.0)
        actual_accel = float(accel.get("actual_accel_mps2", 0.0) or 0.0)
        yaw_accel = abs(float(accel.get("actual_yaw_accel_mps2", 0.0) or 0.0))
        steering = float(control.get("steering_tire_angle_rad", 0.0) or 0.0)
        target_accel = float(control.get("target_accel_mps2", 0.0) or 0.0)

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
            "target_accel": target_accel,
        }
        has_moved = self.vehicle_has_moved or prev_speed > 1.0 or speed > 1.0
        if speed > 5.0:
            if not self.speed_recovery_start_wall_time:
                self.speed_recovery_start_wall_time = now
            elif now - self.speed_recovery_start_wall_time >= 2.0:
                self.active_stuck_spoken = False
                self.stuck_session_start_wall_time = 0.0
                self.stuck_persistent_spoken = False
                self.stop_after_moving_start_wall_time = 0.0
                self.impact_suppress_stuck_until = 0.0
        else:
            self.speed_recovery_start_wall_time = 0.0
        stuck_condition = has_moved and speed_error > 12.0 and speed < 5.0 and actual_accel < 0.05
        stopped_after_moving = has_moved and speed < 1.0
        if stopped_after_moving:
            if not self.stop_after_moving_start_wall_time:
                self.stop_after_moving_start_wall_time = now
        else:
            self.stop_after_moving_start_wall_time = 0.0
        stopped_after_moving_age = (
            now - self.stop_after_moving_start_wall_time
            if self.stop_after_moving_start_wall_time
            else 0.0
        )
        severe_stop_after_moving = (
            has_moved
            and speed < 5.0
            and (
                actual_accel < -1.0
                or yaw_accel > 3.0
                or prev_speed - speed > 8.0
            )
        )
        positive_accel_speed_collapse = (
            has_moved
            and speed_error > 10.0
            and speed < 7.0
            and target_accel > 1.0
            and (prev_speed - speed > 2.0 or speed < 2.0)
        )
        if stuck_condition and not self.stuck_session_start_wall_time:
            self.stuck_session_start_wall_time = now

        event = ""
        reason = ""
        if not prev:
            event = "start"
            reason = "first_sample"
        elif prev and prev_speed - speed > 6.0 and (actual_accel < -0.8 or yaw_accel > 1.2 or speed < 2.0):
            event = "impact_like"
            reason = "rapid_deceleration_or_yaw"
        elif severe_stop_after_moving:
            event = "impact_like"
            reason = "severe_stop_after_moving"
        elif positive_accel_speed_collapse:
            event = "impact_like"
            reason = "positive_accel_speed_collapse"
        elif stuck_condition:
            stuck_age = now - self.stuck_session_start_wall_time
            if (
                not self.stuck_persistent_spoken
                and stuck_age >= 6.0
                and now >= self.impact_suppress_stuck_until
            ):
                event = "stuck_persistent"
                reason = "stuck_session_persistent"
            elif not self.active_stuck_spoken and now >= self.impact_suppress_stuck_until:
                event = "stuck"
                reason = "large_speed_error_low_accel"
        elif (
            stopped_after_moving_age >= 3.0
            and not self.active_stuck_spoken
            and now >= self.impact_suppress_stuck_until
        ):
            event = "stuck"
            reason = "stopped_after_moving"
        elif speed > 1.0 and prev_speed < 1.0:
            if self.vehicle_has_moved:
                event = "restart"
                reason = "speed_recovered_after_stop"
            else:
                event = "start"
                reason = "speed_crossed_1kmh"
        elif driving_state.get("section_entered"):
            event = "section_enter"
            reason = f"section:{driving_state.get('section')}:visit:{driving_state.get('section_visit_count')}"
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

        unstable_sample = actual_accel < -1.5 or yaw_accel > 2.0 or (speed_error > 10.0 and speed < 5.0)
        if event in {"section_enter", "turn_end", "target_speed_reached"} and unstable_sample:
            if severe_stop_after_moving:
                event = "impact_like"
                reason = f"severe_stop_over_{event}:{reason}"
            else:
                self.last_event_data = state
                return "", f"suppress_normal_before_critical:{event}:{reason}"

        self.last_event_data = state
        if speed > 1.0:
            self.vehicle_has_moved = True
        if not event:
            return "", "no_event"
        skip_events = {item.strip() for item in self.args.skip_events.split(",") if item.strip()}
        if event in skip_events:
            return "", f"skip_event:{event}:{reason}"
        if event in ("impact_like", "stuck", "stuck_persistent"):
            if now - self.last_critical_event_wall_time < 4.0:
                return "", f"critical_cooldown:{event}:{reason}"
            self.last_critical_event_wall_time = now
            if event == "impact_like":
                self.impact_suppress_stuck_until = now + 3.0
            if event == "stuck":
                self.active_stuck_spoken = True
            if event == "stuck_persistent":
                self.stuck_persistent_spoken = True
            self.last_spoken_wall_time = now
            return event, reason
        if now - self.last_spoken_wall_time < self.args.min_speak_interval_sec:
            return "", f"min_interval:{event}:{reason}"
        self.last_spoken_wall_time = now
        return event, reason

    def generate_once(self, image, image_stamp, odom, control, accel, vehicle_history) -> None:
        started = time.perf_counter()
        generation_wall_time = time.time()
        try:
            data = self.vehicle_data(image_stamp, odom, control, accel)
            speed = float(data.get("ego", {}).get("speed_kmh", 0.0) or 0.0)
            if speed > 1.0:
                self.vehicle_has_moved = True
            data["vehicle_has_moved"] = self.vehicle_has_moved
            course_context = lookup_course_context(self.course_context, data)
            driving_state = build_driving_state(data, vehicle_history, course_context)
            driving_state = self.annotate_course_progress(data, driving_state)
            if course_context:
                data["course_context"] = course_context
            data["driving_state"] = driving_state
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
            visual_summary = {}
            ambient_context = []
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
                            "Mention road direction, barriers, buildings, sky, or course if changed. "
                            "Use cranes or scaffolding only when clearly visible."
                        )
                    else:
                        visual_prompt = (
                            "Describe this driving scene briefly. "
                            "Mention visible road, sky, barriers, buildings, or course. "
                            "Use cranes or scaffolding only when clearly visible."
                        )
                if self.args.template_style == "vlm_tags":
                    visual_prompt = (
                        "You are labeling a vehicle-mounted camera image for autonomous driving commentary. "
                        "Return only minified JSON with these keys: "
                        "road_direction, curve, barrier, sky, buildings, scene, ambient_context, confidence. "
                        "Allowed road_direction values: left, right, straight, unknown. "
                        "Allowed curve values: none, gentle, sharp, unknown. "
                        "barrier, sky, buildings are booleans. "
                        "scene is one short snake_case label. "
                        "ambient_context is an array using only these labels when visible: "
                        "urban_buildings, blue_white_barrier, open_sky, signboard, narrow_sky_track. "
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
                        "num_predict": 120 if self.args.template_style == "vlm_tags" else 60,
                        "temperature": 0.0 if self.args.template_style == "vlm_tags" else 0.1,
                    },
                }
                t0 = time.perf_counter()
                visual = ollama_chat(
                    self.args.ollama_base_url,
                    visual_payload,
                    timeout=self.ollama_timeout_sec(started),
                )
                self.check_generation_deadline(started, "vision")
                initial_visual_tags = parse_jsonish(visual)
                if (
                    "moondream" in self.args.vision_model.lower()
                    and visual_output_needs_fallback(visual, initial_visual_tags)
                ):
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
                    visual = ollama_chat(
                        self.args.ollama_base_url,
                        fallback_chat_payload,
                        timeout=self.ollama_timeout_sec(started),
                    )
                    self.check_generation_deadline(started, "vision_fallback_chat")
                    if not visual.strip():
                        fallback_generate_payload = {
                            "model": self.args.vision_model,
                            "prompt": fallback_prompt,
                            "images": [current_image_b64],
                            "keep_alive": self.args.ollama_keep_alive,
                            "options": {"num_predict": 50, "temperature": 0.0},
                        }
                        visual = ollama_generate(
                            self.args.ollama_base_url,
                            fallback_generate_payload,
                            timeout=self.ollama_timeout_sec(started),
                        )
                        self.check_generation_deadline(started, "vision_fallback_generate")
                vision_sec = time.perf_counter() - t0
                visual_tags = parse_jsonish(visual)
                if visual_tags:
                    summary = visual_words_summary(visual_tags, visual)
                    visual_tags["visible_words"] = summary["visible_words"]
                    visual_summary = summary
                else:
                    visual_summary = visual_words_summary({}, visual)
                if self.args.template_style == "vlm_tags":
                    ambient_context = derive_ambient_context(visual_tags, visual)
                    if not ambient_context:
                        ambient_context = parse_ambient_context(self.args.default_ambient_context)
                    ambient_context = [
                        tag
                        for tag in ambient_context
                        if not (tag == "blue_white_barrier" and tag in self.mentioned_ambient_tags)
                    ]
                    visual_tags["ambient_context"] = ambient_context
                elif visual_summary.get("visible_words"):
                    visual_tags["visible_words"] = visual_summary["visible_words"]
                self.previous_vlm_image_b64 = current_image_b64
                self.previous_vlm_image = processed.copy()

            previous_commentary = self.last_commentary
            previous_visual = self.last_visual_description
            recent_commentaries = list(self.recent_commentaries[-6:])
            vehicle_summary = summarize_vehicle_history(vehicle_history)
            topic_summary = recent_topic_summary(recent_commentaries)

            if self.args.commentary_mode == "template":
                commentary_type = "reflex"
                t1 = time.perf_counter()
                commentary = template_commentary(visual, data, self.args.template_style, event_type)
                text_sec = time.perf_counter() - t1
            else:
                commentary_types = [
                    "live_call",
                    "why",
                    "watch_point",
                    "sport_commentary",
                    "ambient",
                    "recovery",
                ]
                commentary_type = commentary_types[self.index % len(commentary_types)]
                if data.get("driving_state", {}).get("phase") == "stuck":
                    t1 = time.perf_counter()
                    commentary_type = "recovery"
                    commentary = template_commentary(visual, data, "event", "stuck")
                    text_sec = time.perf_counter() - t1
                elif self.args.template_style == "vlm_tags":
                    t1 = time.perf_counter()
                    commentary = tagged_race_commentary(
                        visual_tags,
                        ambient_context,
                        data,
                        commentary_type,
                        recent_commentaries,
                    )
                    text_sec = time.perf_counter() - t1
                else:
                    persona_instruction = persona_profile(self.args.persona_style)["prompt"]
                    prompt = f"""
あなたは自動運転AIチャレンジの走行を実況するレース解説者です。
舞台はCity Circuit Tokyo Bayの屋外SKY TRACKです。ただし固有名詞はたまにだけ使います。
City Circuit Tokyo Bayは施設名です。都市の中を走っているとは言わず、屋外カートコースをEVカートが走行している前提で話してください。
カメラ画像全体と vehicle_data を見て、自然な実況にしてください。
景色と走行状態は同列に扱い、visual_summary.salient_words の景物が自然に合うときだけ短く添えてください。
建物や空は景色です。車両が建物を抜ける・建物から出る・建物を通過するとは言わないでください。
建物、ビル、クレーン、看板、空が前方やコースの向こうに見える表現は使ってよいです。
	ただし具体的な景物名は visual_summary.salient_words にある場合だけ使ってください。salient_words が空なら景物名を推測しないでください。
	青白いバリアはこのコースに常にあるため、接近・接触・ライン取りに関係しない限り主題にしないでください。
	24〜42文字の日本語1文だけを返してください。説明や箇条書きは禁止です。
	「直線で走っている」「建物が見える」だけの観察で終わらせず、景色か走行状態をもう一要素だけ添えてください。
	同じ言い回し、自己紹介、数値羅列、AWSIM/RViz/UIへの言及は禁止です。
「重視された」のような受け身表現ではなく、「見てた」「狙ってる」「つないでる」のように自然に話してください。
VLM発話は実音声まで数秒遅れるため、「今から始めます」ではなく「今の判断」「ここは」という短い状態描写や振り返りを優先してください。「少し前」は使わないでください。
recent_topic_summary の avoid_topics と同じ話題はできるだけ避けてください。
{persona_instruction}
禁止フレーズは「助手席AI」「最近のコース」「聞こえます」「安全に」「走行中です」「コースを抜けながら」「ビルの間から」「レースは」「ビルを通過」「建物の間を通り抜け」「建物の間を通過」「建物の間を通る」「建物から抜け」「建物を抜け」「建物を通る」「背景の建物から」「直線の後ろ」「速度を上昇」「重視」「重きを置く」「重点を置く」「横目」「自動運転カート」「カートは」です。
「後ろに建物が見えます」のように、背景として後方の景色を添える表現は使ってよいです。

次は良い書き方の方向です。同じ文をそのまま返さず、状況に合わせて言い換えてください。
コース奥の建物を背景に、右コーナーへ入ってる。
奥のクレーンが見えて、ここは舵を落ち着かせる。
開けた空の下、短い直線へつなぐ。
看板が奥に見えて、出口で前へ伸ばしてる。
減速を入れて、向き変えに備えてる。

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

visual_summary:
{json.dumps(visual_summary, ensure_ascii=False)}

vehicle_summary:
{json.dumps(vehicle_summary, ensure_ascii=False)}

vehicle_data:
{json.dumps(data, ensure_ascii=False)}

course_context:
{json.dumps(data.get("course_context", {}), ensure_ascii=False)}

driving_state:
{json.dumps(data.get("driving_state", {}), ensure_ascii=False)}

recent_topic_summary:
{json.dumps(topic_summary, ensure_ascii=False)}
""".strip()
                    text_options = {"num_predict": 80, "temperature": 0.35}
                    system_prompt = "日本語の実況1文だけを返す。説明、前置き、箇条書きは禁止。"
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
                    raw_commentary = ollama_chat(
                        self.args.ollama_base_url,
                        text_payload,
                        timeout=self.ollama_timeout_sec(started),
                    )
                    self.check_generation_deadline(started, "text")
                    commentary = clean_commentary(raw_commentary, self.args.max_commentary_chars, 1)
                    if (
                        commentary_needs_retry(commentary)
                        or commentary_uses_unavailable_visual_word(commentary, visual_summary)
                        or commentary_is_low_information(commentary)
                        or commentary_repeats_any(commentary, recent_commentaries)
                        or commentary_repeats_recent_phrase(commentary, recent_commentaries)
                    ):
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
                                        "EVカートの走行実況を日本語1文で返してください。"
                                        "18〜34文字。説明、箇条書き、数値羅列は禁止。"
                                        "敬語ではなくカジュアルなタメ口。"
                                        "重視された、のような受け身は使わない。"
                                        "実音声は少し遅れるので、予告より状態描写か振り返り。"
                                        "少し前、とは言わない。"
                                        "風景語があれば、自然に合うときだけ建物、ビル、クレーン、空、看板を短く添える。"
                                        "ただし具体的な景物名はVLM語彙のsalient_wordsにある場合だけ使う。"
	                                        "salient_wordsが空なら景物名を推測しない。"
	                                        "青白いバリアは常設なので主題にしない。"
	                                        "景色と車両挙動のどちらか自然な方を選ぶ。風景だけの説明にはしない。"
	                                        "直線で走っている、建物が見える、だけの短文は禁止。"
	                                        "直近の話題カテゴリと同じ話題はできるだけ避ける。"
                                        "禁止: 助手席AI、最近のコース、聞こえます、安全に、走行中です、自動運転カート、"
                                        "コースを抜けながら、ビルの間から、レースは、ビルを通過、"
                                        "建物から抜け、建物を抜け、背景の建物から、直線の後ろ、"
                                        "前方を抜け、速度を上昇、目標速度へ向けて加速。"
                                        "ただし、後ろに見える背景を短く添えるのは可。\n\n"
                                        "例文はそのまま使わず、状況に合わせて言い換える。\n"
                                        "良い方向: 奥の建物を背景に、右コーナーへ入ってる。\n"
                                        "良い方向: 開けた空の下、短い直線へつなぐ。\n\n"
                                        f"最近の実況: {json.dumps(recent_commentaries, ensure_ascii=False)}\n"
                                        f"最近の話題: {json.dumps(topic_summary, ensure_ascii=False)}\n"
                                        f"実況タイプ: {commentary_type}\n"
                                        f"VLM変化: {visual}\n"
                                        f"VLM語彙: {json.dumps(visual_summary, ensure_ascii=False)}\n"
                                        f"車両履歴: {json.dumps(vehicle_summary, ensure_ascii=False)}\n"
                                        f"車両状態: {json.dumps(data, ensure_ascii=False)}\n"
                                        f"コース特徴: {json.dumps(data.get('course_context', {}), ensure_ascii=False)}\n"
                                        f"走行フェーズ: {json.dumps(data.get('driving_state', {}), ensure_ascii=False)}"
                                    ),
                                },
                            ],
                            "options": {"num_predict": 45, "temperature": 0.2},
                        }
                        commentary = clean_commentary(
                            ollama_chat(
                                self.args.ollama_base_url,
                                retry_payload,
                                timeout=self.ollama_timeout_sec(started),
                            ),
                            self.args.max_commentary_chars,
                            1,
                        )
                        self.check_generation_deadline(started, "text_retry")
                    if (
                        commentary_needs_retry(commentary)
                        or commentary_uses_unavailable_visual_word(commentary, visual_summary)
                        or commentary_is_low_information(commentary)
                    ):
                        commentary = race_commentary_fallback(visual, data, recent_commentaries)
                    if commentary_repeats_any(commentary, recent_commentaries) or commentary_repeats_recent_phrase(commentary, recent_commentaries):
                        commentary = race_commentary_fallback(visual, data, recent_commentaries)
                    text_sec = time.perf_counter() - t1

            commentary = maybe_add_ambient_prefix(
                commentary,
                ambient_context,
                self.index,
                commentary_type,
            )
            commentary = apply_persona_layer(
                commentary,
                commentary_type,
                self.args.max_commentary_chars,
                self.args.persona_style,
            )
            template_sentences = 2 if self.args.template_style in ("event", "scenery") else 1
            commentary = clean_commentary(commentary, self.args.max_commentary_chars, template_sentences)
            commentary = final_non_repeating_commentary(
                commentary,
                visual,
                visual_summary,
                data,
                recent_commentaries,
                self.args.max_commentary_chars,
                self.args.persona_style,
                commentary_type,
            )
            commentary = maybe_blend_scene_phrase(
                commentary,
                visual_summary,
                recent_commentaries,
                self.args.max_commentary_chars,
                commentary_type,
                self.index,
            )

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
                "visual_summary": visual_summary,
                "ambient_context": ambient_context,
                "vehicle_data": data,
                "commentary": commentary,
                "commentary_mode": self.args.commentary_mode,
                "commentary_trigger": self.args.commentary_trigger,
                "event_type": event_type or None,
                "event_reason": event_reason or None,
                "commentary_type": commentary_type,
                "persona_style": self.args.persona_style,
                "vehicle_summary": vehicle_summary,
                "recent_topic_summary": recent_topic_summary(self.recent_commentaries[-6:]),
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
            for tag in ambient_context:
                if tag == "blue_white_barrier" and "バリア" in commentary:
                    self.mentioned_ambient_tags.add(tag)
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
    parser.add_argument("--skip-events", default=os.getenv("SKIP_EVENTS", ""))
    parser.add_argument("--min-speak-interval-sec", type=float, default=float(os.getenv("MIN_SPEAK_INTERVAL_SEC", "4.0")))
    parser.add_argument("--template-style", default=os.getenv("TEMPLATE_STYLE", "normal"), choices=["normal", "short", "event", "event_fast", "scenery", "vlm_tags"])
    parser.add_argument("--persona-style", default=os.getenv("PERSONA_STYLE", "passenger_casual_light"), choices=sorted(PERSONA_PROFILES))
    parser.add_argument("--default-ambient-context", default=os.getenv("DEFAULT_AMBIENT_CONTEXT", ""))
    parser.add_argument("--course-context", default=os.getenv("COURSE_CONTEXT", ""))
    parser.add_argument("--max-commentary-chars", type=int, default=int(os.getenv("MAX_COMMENTARY_CHARS", "45")))
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-keep-alive", default=os.getenv("OLLAMA_KEEP_ALIVE", "30m"))
    parser.add_argument("--ollama-timeout-sec", type=float, default=float(os.getenv("OLLAMA_TIMEOUT_SEC", "180")))
    parser.add_argument("--generation-deadline-sec", type=float, default=float(os.getenv("GENERATION_DEADLINE_SEC", "0")))
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
