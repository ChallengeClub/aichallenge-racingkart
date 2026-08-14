#!/usr/bin/env python3
"""Forward immediate and VLM recap commentary JSONL streams to AITuberKit."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


CRITICAL_EVENTS = {"impact_like", "stuck", "stuck_persistent", "restart"}
RECOVERY_EVENTS = {"recovery"}

SPEECH_TYPE_DEFAULTS = {
    "critical": {"priority": 100, "expireSec": 3.0},
    "stuck": {"priority": 100, "expireSec": 6.0},
    "recovery": {"priority": 100, "expireSec": 6.0},
    "course_note": {"priority": 60, "expireSec": 6.0},
    "vlm_recap": {"priority": 50, "expireSec": 6.0},
    "immediate": {"priority": 30, "expireSec": 4.0},
    "normal": {"priority": 20, "expireSec": 5.0},
}


def estimate_speech_duration_sec(text: str) -> float:
    """Rough guard for AITuberKit/VOICEVOX queue pressure."""
    text_len = len(text.strip())
    return min(max(1.2, 0.17 * text_len + 0.6), 5.8)


def infer_speech_type(lane: str, data: dict) -> str:
    explicit = data.get("speechType") or data.get("speech_type")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip().lower()
    ev = event_type(data)
    if ev in ("stuck", "stuck_persistent"):
        return "stuck"
    if ev in RECOVERY_EVENTS or data.get("commentary_type") == "recovery":
        return "recovery"
    if ev in CRITICAL_EVENTS:
        return "critical"
    if data.get("commentary_type") == "course_note" or lane == "course_note":
        return "course_note"
    if lane == "vlm_recap":
        return "vlm_recap"
    if lane == "immediate":
        return "immediate"
    return "normal"


def first_value(*values):
    for value in values:
        if value is not None:
            return value
    return None


def nested(mapping: dict, *keys: str):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def compact_driving_context(lane: str, data: dict, speech_type: str) -> dict:
    vehicle_data = data.get("vehicle_data") or data.get("vehicleData") or {}
    vehicle_summary = data.get("vehicle_summary") or data.get("vehicleSummary") or {}
    visual_summary = data.get("visual_summary") or data.get("visualSummary")

    context = {
        "lane": lane,
        "speechType": speech_type,
    }
    for key in ("index", "image_stamp_sec", "event_type", "event_reason", "commentary_type"):
        if data.get(key) is not None:
            context[key] = data.get(key)

    if isinstance(vehicle_data, dict):
        values = {
            "speed_kmh": first_value(nested(vehicle_data, "ego", "speed_kmh"), vehicle_data.get("speed_kmh")),
            "target_speed_kmh": first_value(
                nested(vehicle_data, "control_cmd", "target_speed_kmh"),
                vehicle_data.get("target_speed_kmh"),
            ),
            "phase": first_value(nested(vehicle_data, "driving_state", "phase"), vehicle_data.get("phase")),
            "vehicle_has_moved": vehicle_data.get("vehicle_has_moved"),
            "actual_accel_mps2": nested(vehicle_data, "acceleration", "actual_accel_mps2"),
        }
        context.update({key: value for key, value in values.items() if value is not None})

    if isinstance(vehicle_summary, dict):
        values = {
            "speed_kmh": first_value(context.get("speed_kmh"), vehicle_summary.get("speed_kmh_end")),
            "target_speed_kmh": first_value(context.get("target_speed_kmh"), vehicle_summary.get("target_speed_kmh_end")),
            "motion_trend": vehicle_summary.get("motion_trend"),
            "turning": vehicle_summary.get("turning"),
        }
        context.update({key: value for key, value in values.items() if value is not None})

    if isinstance(visual_summary, str) and visual_summary.strip():
        context["visual_summary"] = visual_summary.strip()
    return context


def speech_metadata(lane: str, data: dict) -> dict:
    speech_type = infer_speech_type(lane, data)
    defaults = SPEECH_TYPE_DEFAULTS.get(speech_type, SPEECH_TYPE_DEFAULTS["normal"])
    metadata = {
        "source": "vlm_bridge",
        "lane": lane,
        "speechType": speech_type,
        "priority": data.get("priority", defaults["priority"]),
        "expireSec": data.get("expireSec", data.get("expire_sec", defaults["expireSec"])),
    }
    for key in (
        "index",
        "image_stamp_sec",
        "generation_wall_time_sec",
        "audio_ready_wall_time_sec",
        "event_type",
        "event_reason",
        "commentary_type",
        "commentary_mode",
        "commentary_trigger",
    ):
        if key in data and data.get(key) is not None:
            metadata[key] = data.get(key)
    driving_context = compact_driving_context(lane, data, speech_type)
    if len(driving_context) > 2:
        metadata["drivingContext"] = driving_context
    return metadata


def is_high_priority_speech(lane: str, data: dict) -> bool:
    return infer_speech_type(lane, data) in {"critical", "stuck", "recovery"}


def post_message(endpoint: str, text: str, metadata: dict) -> dict:
    payload = json.dumps(
        {
            "message": text,
            "text": text,
            "speechType": metadata["speechType"],
            "priority": metadata["priority"],
            "expireSec": metadata["expireSec"],
            "source": metadata["source"],
            "metadata": metadata,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"raw_response": raw}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST failed: {exc.code} {body}") from exc


def extract_text(data: dict) -> str:
    return str(
        data.get("commentary")
        or data.get("text")
        or data.get("message")
        or data.get("speech_text")
        or ""
    ).strip()


def normalize_text(text: str) -> str:
    return "".join(str(text).split())


def add_recap_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    stripped = text.strip()
    if stripped.startswith(("少し前", "今の", "ここは", "さっき")):
        return stripped
    return f"{prefix}{stripped}"


def suppress_repeated_static_scenery(text: str, sent_static_tags: set[str]) -> str:
    if "blue_white_barrier" not in sent_static_tags:
        return text
    for phrase in ("青白いバリア沿い、", "青白いバリア沿いで、", "青白いバリアが続く、"):
        text = text.replace(phrase, "")
    return text.strip()


def clamp_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip(" 、,。")
    if clipped and clipped[-1] not in ("。", "！", "!", "？", "?"):
        clipped += "。"
    return clipped


def event_key(data: dict) -> str | None:
    event_type = data.get("event_type")
    event_reason = data.get("event_reason")
    if not event_type:
        return None
    return f"{event_type}:{event_reason or ''}"


def event_type(data: dict) -> str:
    return str(data.get("event_type") or "")


def item_priority(item: dict) -> tuple[int, float]:
    data = item["data"]
    if is_high_priority_speech(item["lane"], data):
        return (0, item["line_seen_wall_time_sec"])
    if item["lane"] == "immediate":
        return (1, item["line_seen_wall_time_sec"])
    return (2, item["line_seen_wall_time_sec"])


class Tailer:
    def __init__(self, path: Path, lane: str, priority: int, from_start: bool) -> None:
        self.path = path
        self.lane = lane
        self.priority = priority
        self.from_start = from_start
        self.file = None
        self.started = False

    def open_if_ready(self) -> None:
        if self.file is not None or not self.path.exists():
            return
        self.file = self.path.open("r", encoding="utf-8")
        if not self.from_start:
            self.file.seek(0, 2)
        self.started = True

    def read_new(self) -> list[dict]:
        self.open_if_ready()
        if self.file is None:
            return []
        out = []
        while True:
            raw = self.file.readline()
            if not raw:
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = extract_text(data)
            if not text:
                continue
            out.append(
                {
                    "lane": self.lane,
                    "priority": self.priority,
                    "data": data,
                    "text": text,
                    "line_seen_wall_time_sec": time.time(),
                }
            )
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--immediate-jsonl", required=True)
    parser.add_argument("--recap-jsonl", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8018/send_message")
    parser.add_argument("--from-start", action="store_true")
    parser.add_argument("--poll-sec", type=float, default=0.05)
    parser.add_argument("--min-send-interval-sec", type=float, default=1.2)
    parser.add_argument("--recap-min-send-interval-sec", type=float, default=5.0)
    parser.add_argument("--recap-max-age-sec", type=float, default=8.0)
    parser.add_argument("--immediate-duplicate-cooldown-sec", type=float, default=12.0)
    parser.add_argument("--recap-duplicate-cooldown-sec", type=float, default=18.0)
    parser.add_argument("--immediate-event-cooldown-sec", type=float, default=7.0)
    parser.add_argument("--critical-event-cooldown-sec", type=float, default=3.0)
    parser.add_argument("--critical-duplicate-cooldown-sec", type=float, default=4.0)
    parser.add_argument("--critical-min-send-interval-sec", type=float, default=0.0)
    parser.add_argument("--non-critical-hold-sec", type=float, default=0.8)
    parser.add_argument("--immediate-max-queued-sec", type=float, default=0.5)
    parser.add_argument("--critical-max-queued-sec", type=float, default=5.0)
    parser.add_argument("--recap-after-any-send-sec", type=float, default=2.5)
    parser.add_argument("--recap-max-queued-sec", type=float, default=2.0)
    parser.add_argument("--recap-reserve-interval-sec", type=float, default=0.0)
    parser.add_argument("--recap-reserve-max-age-sec", type=float, default=18.0)
    parser.add_argument("--recap-reserve-max-queued-sec", type=float, default=1.5)
    parser.add_argument("--recap-block-after-critical-sec", type=float, default=0.0)
    parser.add_argument("--recap-block-after-immediate-sec", type=float, default=0.0)
    parser.add_argument("--recap-max-chars", type=int, default=0)
    parser.add_argument("--latest-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recap-prefix", default="")
    parser.add_argument("--idle-timeout-sec", type=float, default=0.0)
    args = parser.parse_args()

    tailers = [
        Tailer(Path(args.immediate_jsonl), "immediate", 0, args.from_start),
        Tailer(Path(args.recap_jsonl), "vlm_recap", 1, args.from_start),
    ]
    pending: list[dict] = []
    sent = 0
    dropped = 0
    started = time.time()
    last_send_wall = 0.0
    last_recap_send_wall = 0.0
    last_critical_seen_wall = 0.0
    last_critical_send_wall = 0.0
    last_immediate_seen_wall = 0.0
    last_immediate_send_wall = 0.0
    virtual_speech_available_wall = 0.0
    last_text_by_lane: dict[tuple[str, str], float] = {}
    last_immediate_event: dict[str, float] = {}
    sent_static_scenery_tags: set[str] = set()
    last_activity_monotonic = time.monotonic()

    while True:
        for tailer in tailers:
            new_items = tailer.read_new()
            if new_items:
                seen_wall = time.time()
                for new_item in new_items:
                    if new_item["lane"] == "immediate":
                        last_immediate_seen_wall = seen_wall
                    if is_high_priority_speech(new_item["lane"], new_item["data"]):
                        last_critical_seen_wall = seen_wall
                pending.extend(new_items)
                if args.latest_only:
                    latest_critical_by_event: dict[str, dict] = {}
                    latest_immediate = None
                    latest_recap = None
                    for pending_item in pending:
                        pending_event = event_type(pending_item["data"])
                        if is_high_priority_speech(pending_item["lane"], pending_item["data"]):
                            latest_critical_by_event[pending_event or infer_speech_type(pending_item["lane"], pending_item["data"])] = pending_item
                        elif pending_item["lane"] == "immediate":
                            latest_immediate = pending_item
                        else:
                            latest_recap = pending_item
                    pending = list(latest_critical_by_event.values())
                    if latest_immediate is not None:
                        pending.append(latest_immediate)
                    if latest_recap is not None:
                        pending.append(latest_recap)
                last_activity_monotonic = time.monotonic()

        if not pending:
            any_started = any(t.started for t in tailers)
            if args.idle_timeout_sec and any_started and time.monotonic() - last_activity_monotonic > args.idle_timeout_sec:
                break
            if args.idle_timeout_sec and not any_started and time.time() - started > args.idle_timeout_sec:
                break
            time.sleep(args.poll_sec)
            continue

        now_wall = time.time()
        reserved_recap_index = None
        if args.recap_reserve_interval_sec and now_wall - last_recap_send_wall >= args.recap_reserve_interval_sec:
            has_critical = any(is_high_priority_speech(p["lane"], p["data"]) for p in pending)
            if not has_critical and now_wall - last_send_wall >= args.recap_after_any_send_sec:
                recap_candidates = []
                for index, pending_item in enumerate(pending):
                    if pending_item["lane"] != "vlm_recap":
                        continue
                    generation_wall = pending_item["data"].get("generation_wall_time_sec") or pending_item["data"].get("audio_ready_wall_time_sec")
                    if generation_wall is not None and now_wall - float(generation_wall) > args.recap_reserve_max_age_sec:
                        continue
                    recap_candidates.append((index, pending_item["line_seen_wall_time_sec"]))
                if recap_candidates:
                    reserved_recap_index = max(recap_candidates, key=lambda row: row[1])[0]

        if reserved_recap_index is not None:
            item = pending.pop(reserved_recap_index)
            item["recap_reserved"] = True
        else:
            pending.sort(key=item_priority)
            item = pending.pop(0)
        data = item["data"]
        lane = item["lane"]
        ev = event_type(data)
        is_critical = is_high_priority_speech(lane, data)
        is_reserved_recap = bool(item.get("recap_reserved"))
        if not is_critical and args.non_critical_hold_sec > 0:
            item_age_sec = now_wall - float(item.get("line_seen_wall_time_sec") or now_wall)
            if item_age_sec < args.non_critical_hold_sec:
                pending.append(item)
                time.sleep(min(args.poll_sec, args.non_critical_hold_sec - item_age_sec))
                continue
        text = item["text"]
        text = suppress_repeated_static_scenery(text, sent_static_scenery_tags)
        if lane == "vlm_recap":
            text = clamp_text(text, args.recap_max_chars)
        text_key = normalize_text(text)
        queued_sec = max(0.0, virtual_speech_available_wall - now_wall)

        duplicate_cooldown = (
            args.critical_duplicate_cooldown_sec
            if is_critical
            else args.immediate_duplicate_cooldown_sec
            if lane == "immediate"
            else args.recap_duplicate_cooldown_sec
        )
        last_same_text = last_text_by_lane.get((lane, text_key))
        if duplicate_cooldown and last_same_text is not None and now_wall - last_same_text < duplicate_cooldown:
            dropped += 1
            print(
                json.dumps(
                    {
                        "event": "drop",
                        "reason": "duplicate_text_cooldown",
                        "lane": lane,
                        "index": data.get("index"),
                        "cooldown_sec": duplicate_cooldown,
                        "age_sec": now_wall - last_same_text,
                        "text": text,
                        "wall_time_sec": now_wall,
                        "sent": sent,
                        "dropped": dropped,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        if lane == "immediate":
            key = event_key(data)
            last_same_event = last_immediate_event.get(key) if key else None
            event_cooldown = args.critical_event_cooldown_sec if is_critical else args.immediate_event_cooldown_sec
            if (
                key
                and event_cooldown
                and last_same_event is not None
                and now_wall - last_same_event < event_cooldown
            ):
                dropped += 1
                print(
                    json.dumps(
                        {
                            "event": "drop",
                            "reason": "immediate_event_cooldown",
                            "lane": lane,
                            "index": data.get("index"),
                            "event_key": key,
                            "cooldown_sec": event_cooldown,
                            "age_sec": now_wall - last_same_event,
                            "text": text,
                            "wall_time_sec": now_wall,
                            "sent": sent,
                            "dropped": dropped,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue

        if lane == "immediate":
            max_queued = args.critical_max_queued_sec if is_critical else args.immediate_max_queued_sec
            if max_queued and queued_sec > max_queued:
                dropped += 1
                print(
                    json.dumps(
                        {
                            "event": "drop",
                            "reason": "immediate_queue_budget",
                            "lane": lane,
                            "index": data.get("index"),
                            "event_type": ev or None,
                            "queued_sec": queued_sec,
                            "max_queued_sec": max_queued,
                            "image_stamp_sec": data.get("image_stamp_sec"),
                            "text": text,
                            "wall_time_sec": now_wall,
                            "sent": sent,
                            "dropped": dropped,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue

        min_send_interval = args.critical_min_send_interval_sec if is_critical else args.min_send_interval_sec
        if min_send_interval and not is_reserved_recap and now_wall - last_send_wall < min_send_interval:
            dropped += 1
            print(
                json.dumps(
                    {
                        "event": "drop",
                        "reason": "min_send_interval",
                        "lane": lane,
                        "index": data.get("index"),
                        "event_type": ev or None,
                        "min_send_interval_sec": min_send_interval,
                        "image_stamp_sec": data.get("image_stamp_sec"),
                        "text": text,
                        "wall_time_sec": now_wall,
                        "sent": sent,
                        "dropped": dropped,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            continue

        if lane == "vlm_recap":
            critical_block_age = now_wall - max(last_critical_seen_wall, last_critical_send_wall)
            if args.recap_block_after_critical_sec and critical_block_age < args.recap_block_after_critical_sec:
                dropped += 1
                print(
                    json.dumps(
                        {
                            "event": "drop",
                            "reason": "recap_block_after_critical",
                            "lane": lane,
                            "index": data.get("index"),
                            "age_sec": critical_block_age,
                            "block_sec": args.recap_block_after_critical_sec,
                            "text": text,
                            "wall_time_sec": now_wall,
                            "sent": sent,
                            "dropped": dropped,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            immediate_block_age = now_wall - max(last_immediate_seen_wall, last_immediate_send_wall)
            if args.recap_block_after_immediate_sec and immediate_block_age < args.recap_block_after_immediate_sec:
                dropped += 1
                print(
                    json.dumps(
                        {
                            "event": "drop",
                            "reason": "recap_block_after_immediate",
                            "lane": lane,
                            "index": data.get("index"),
                            "age_sec": immediate_block_age,
                            "block_sec": args.recap_block_after_immediate_sec,
                            "text": text,
                            "wall_time_sec": now_wall,
                            "sent": sent,
                            "dropped": dropped,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            recap_max_queued_sec = args.recap_reserve_max_queued_sec if is_reserved_recap else args.recap_max_queued_sec
            if recap_max_queued_sec and queued_sec > recap_max_queued_sec:
                dropped += 1
                print(
                    json.dumps(
                        {
                            "event": "drop",
                            "reason": "recap_queue_budget",
                            "lane": lane,
                            "index": data.get("index"),
                            "queued_sec": queued_sec,
                            "max_queued_sec": recap_max_queued_sec,
                            "recap_reserved": is_reserved_recap,
                            "text": text,
                            "wall_time_sec": now_wall,
                            "sent": sent,
                            "dropped": dropped,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            if args.recap_after_any_send_sec and now_wall - last_send_wall < args.recap_after_any_send_sec:
                dropped += 1
                print(
                    json.dumps(
                        {
                            "event": "drop",
                            "reason": "recap_after_any_send",
                            "lane": lane,
                            "index": data.get("index"),
                            "age_sec": now_wall - last_send_wall,
                            "min_age_sec": args.recap_after_any_send_sec,
                            "text": text,
                            "wall_time_sec": now_wall,
                            "sent": sent,
                            "dropped": dropped,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            generation_wall = data.get("generation_wall_time_sec") or data.get("audio_ready_wall_time_sec")
            recap_max_age_sec = args.recap_reserve_max_age_sec if is_reserved_recap else args.recap_max_age_sec
            if generation_wall is not None and now_wall - float(generation_wall) > recap_max_age_sec:
                dropped += 1
                print(
                    json.dumps(
                        {
                            "event": "drop",
                            "reason": "recap_stale",
                            "lane": lane,
                            "index": data.get("index"),
                            "age_sec": now_wall - float(generation_wall),
                            "max_age_sec": recap_max_age_sec,
                            "recap_reserved": is_reserved_recap,
                            "text": text,
                            "wall_time_sec": now_wall,
                            "sent": sent,
                            "dropped": dropped,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            if now_wall - last_recap_send_wall < args.recap_min_send_interval_sec:
                dropped += 1
                print(
                    json.dumps(
                        {
                            "event": "drop",
                            "reason": "recap_min_interval",
                            "lane": lane,
                            "index": data.get("index"),
                            "text": text,
                            "wall_time_sec": now_wall,
                            "sent": sent,
                            "dropped": dropped,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            text = add_recap_prefix(text, args.recap_prefix)

        post_start_wall = time.time()
        metadata = speech_metadata(lane, data)
        post_response = post_message(args.endpoint, text, metadata)
        post_done_wall = time.time()
        last_send_wall = post_done_wall
        last_text_by_lane[(lane, text_key)] = post_done_wall
        if lane == "immediate":
            key = event_key(data)
            if key:
                last_immediate_event[key] = post_done_wall
            last_immediate_send_wall = post_done_wall
            if is_critical:
                last_critical_send_wall = post_done_wall
        if lane == "vlm_recap":
            last_recap_send_wall = post_done_wall
        if "青白いバリア" in text:
            sent_static_scenery_tags.add("blue_white_barrier")
        virtual_speech_available_wall = max(post_done_wall, virtual_speech_available_wall) + estimate_speech_duration_sec(text)
        sent += 1
        print(
            json.dumps(
                {
                    "event": "sent",
                    "sent": sent,
                    "dropped": dropped,
                    "lane": lane,
                    "index": data.get("index"),
                    "event_type": ev or None,
                    "speechType": metadata["speechType"],
                    "speechPriority": metadata["priority"],
                    "speechExpireSec": metadata["expireSec"],
                    "image_stamp_sec": data.get("image_stamp_sec"),
                    "generation_wall_time_sec": data.get("generation_wall_time_sec"),
                    "audio_ready_wall_time_sec": data.get("audio_ready_wall_time_sec"),
                    "latency_sec": data.get("latency_sec"),
                    "line_seen_wall_time_sec": item["line_seen_wall_time_sec"],
                    "estimated_queue_sec_before_send": queued_sec,
                    "estimated_speech_duration_sec": estimate_speech_duration_sec(text),
                    "post_start_wall_time_sec": post_start_wall,
                    "post_done_wall_time_sec": post_done_wall,
                    "post_duration_sec": post_done_wall - post_start_wall,
                    "aituber_event_id": post_response.get("eventId"),
                    "aituber_post_response": post_response,
                    "recap_reserved": is_reserved_recap,
                    "text": text,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
