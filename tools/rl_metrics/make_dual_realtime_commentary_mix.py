#!/usr/bin/env python3
"""Mix primary driving commentary and secondary scenery commentary."""

from __future__ import annotations

import argparse
import array
import json
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Track:
    role: str
    index: int
    requested_start: float
    play_start: float
    duration: float
    path: Path
    text: str
    generation_offset: float
    ready_offset: float
    total_latency: float
    dropped: bool = False
    dropped_reason: str = ""


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def probe_video_duration(ffmpeg_image: str, video_path: Path) -> float:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{video_path.parent.resolve()}:/work:ro",
        ffmpeg_image,
        "-hide_banner",
        "-i",
        f"/work/{video_path.name}",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("Duration:"):
            value = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
            hh, mm, ss = value.split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    raise RuntimeError(f"failed to probe duration for {video_path}")


def load_role_tracks(role: str, commentary_dir: Path, timeline_start_wall_sec: float) -> list[Track]:
    jsonl_path = commentary_dir / "realtime_commentary.jsonl"
    audio_dir = commentary_dir / "voicevox"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tracks: list[Track] = []
    for row in rows:
        idx = int(row["index"])
        path = audio_dir / f"line_{idx:04d}.wav"
        if not path.exists():
            continue
        generation_wall = float(row.get("generation_wall_time_sec") or row["vehicle_data"]["wall_time_sec"])
        ready_wall = float(row.get("audio_ready_wall_time_sec") or (generation_wall + float(row.get("latency_sec", {}).get("total") or 0.0)))
        generation_offset = generation_wall - timeline_start_wall_sec
        ready_offset = ready_wall - timeline_start_wall_sec
        tracks.append(
            Track(
                role=role,
                index=idx,
                requested_start=ready_offset,
                play_start=ready_offset,
                duration=wav_duration(path),
                path=path,
                text=str(row["commentary"]),
                generation_offset=generation_offset,
                ready_offset=ready_offset,
                total_latency=float(row.get("latency_sec", {}).get("total") or 0.0),
            )
        )
    return tracks


def schedule_tracks(
    primary: list[Track],
    secondary: list[Track],
    duration_sec: float,
    drop_stale_sec: float,
    secondary_guard_sec: float,
    primary_dedupe_window_sec: float,
) -> list[Track]:
    tracks = sorted(primary + secondary, key=lambda item: (item.requested_start, 0 if item.role == "primary" else 1))
    primary_starts = sorted(item.requested_start for item in primary if item.requested_start >= 0.0)
    scheduled: list[Track] = []
    next_available = 0.0
    last_primary_text_time: dict[str, float] = {}
    for track in tracks:
        if track.requested_start < 0.0:
            track.dropped = True
            track.dropped_reason = "before_timeline_start"
            scheduled.append(track)
            continue
        if track.requested_start >= duration_sec:
            track.dropped = True
            track.dropped_reason = "after_timeline_end"
            scheduled.append(track)
            continue
        if track.role == "secondary":
            if track.requested_start - track.generation_offset > drop_stale_sec:
                track.dropped = True
                track.dropped_reason = "stale"
                scheduled.append(track)
                continue
            if track.requested_start < next_available + secondary_guard_sec:
                track.dropped = True
                track.dropped_reason = "primary_busy"
                scheduled.append(track)
                continue
            next_primary = next((start for start in primary_starts if start > track.requested_start), None)
            if next_primary is not None and track.requested_start + track.duration + secondary_guard_sec > next_primary:
                track.dropped = True
                track.dropped_reason = "before_primary"
                scheduled.append(track)
                continue
            track.play_start = track.requested_start
            next_available = max(next_available, track.play_start + track.duration)
            scheduled.append(track)
            continue

        track.play_start = max(track.requested_start, next_available)
        last_same_text = last_primary_text_time.get(track.text)
        if (
            primary_dedupe_window_sec > 0.0
            and last_same_text is not None
            and track.play_start - last_same_text < primary_dedupe_window_sec
        ):
            track.dropped = True
            track.dropped_reason = "duplicate_primary"
            scheduled.append(track)
            continue
        if track.play_start - track.generation_offset > drop_stale_sec:
            track.dropped = True
            track.dropped_reason = "stale"
        elif track.play_start >= duration_sec:
            track.dropped = True
            track.dropped_reason = "after_timeline_end"
        else:
            next_available = track.play_start + track.duration
            last_primary_text_time[track.text] = track.play_start
        scheduled.append(track)
    return scheduled


def mix_tracks(tracks: list[Track], duration_sec: float, output_wav: Path) -> int:
    rate = channels = sampwidth = None
    loaded = []
    for track in tracks:
        if track.dropped:
            continue
        with wave.open(str(track.path), "rb") as wav:
            fmt = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
            if rate is None:
                rate, channels, sampwidth = fmt
            if fmt != (rate, channels, sampwidth):
                raise RuntimeError(f"wav format mismatch: {track.path}")
            data = wav.readframes(wav.getnframes())
        loaded.append((track.play_start, data))
    if rate is None or channels is None or sampwidth is None:
        raise RuntimeError("no audio tracks inside duration")
    if sampwidth != 2:
        raise RuntimeError(f"unsupported sample width: {sampwidth}")
    mixed = array.array("h", [0]) * (int(duration_sec * rate) * channels)
    for start, data in loaded:
        offset = int(start * rate) * channels
        samples = array.array("h")
        samples.frombytes(data)
        size = min(len(samples), len(mixed) - offset)
        for index in range(max(0, size)):
            value = mixed[offset + index] + samples[index]
            mixed[offset + index] = max(-32768, min(32767, value))
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(rate)
        wav.writeframes(mixed.tobytes())
    return len(loaded)


def write_schedule(tracks: list[Track], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for track in tracks:
            file.write(
                json.dumps(
                    {
                        "role": track.role,
                        "index": track.index,
                        "requested_start_sec": round(track.requested_start, 3),
                        "play_start_sec": round(track.play_start, 3),
                        "audio_duration_sec": round(track.duration, 3),
                        "queue_delay_sec": round(track.play_start - track.requested_start, 3),
                        "total_delay_from_generation_sec": round(track.play_start - track.generation_offset, 3),
                        "generation_latency_sec": round(track.total_latency, 3),
                        "dropped": track.dropped,
                        "dropped_reason": track.dropped_reason,
                        "commentary": track.text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def mux_video(ffmpeg_image: str, video_path: Path, audio_path: Path, output_mp4: Path) -> None:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{video_path.parent.resolve()}:/video:ro",
        "-v",
        f"{audio_path.parent.resolve()}:/audio:ro",
        "-v",
        f"{output_mp4.parent.resolve()}:/out",
        ffmpeg_image,
        "-y",
        "-hide_banner",
        "-i",
        f"/video/{video_path.name}",
        "-i",
        f"/audio/{audio_path.name}",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        f"/out/{output_mp4.name}",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--timeline-start-wall-file", type=Path, required=True)
    parser.add_argument("--primary-dir", type=Path, default=None)
    parser.add_argument("--secondary-dir", type=Path, default=None)
    parser.add_argument("--drop-stale-sec", type=float, default=8.0)
    parser.add_argument("--secondary-guard-sec", type=float, default=0.5)
    parser.add_argument("--primary-dedupe-window-sec", type=float, default=10.0)
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--output-mp4", type=Path, required=True)
    parser.add_argument("--schedule-jsonl", type=Path, required=True)
    parser.add_argument("--ffmpeg-image", default="linuxserver/ffmpeg:latest")
    args = parser.parse_args()

    timeline_start_wall_sec = float(args.timeline_start_wall_file.read_text().strip())
    duration_sec = probe_video_duration(args.ffmpeg_image, args.video)
    primary_dir = args.primary_dir or args.run_dir / "realtime_commentary"
    secondary_dir = args.secondary_dir or args.run_dir / "scenery_commentary"
    tracks = schedule_tracks(
        load_role_tracks("primary", primary_dir, timeline_start_wall_sec),
        load_role_tracks("secondary", secondary_dir, timeline_start_wall_sec),
        duration_sec,
        args.drop_stale_sec,
        args.secondary_guard_sec,
        args.primary_dedupe_window_sec,
    )
    count = mix_tracks(tracks, duration_sec, args.output_wav)
    write_schedule(tracks, args.schedule_jsonl)
    mux_video(args.ffmpeg_image, args.video, args.output_wav, args.output_mp4)
    print(
        f"wrote {args.output_mp4} ({count} tracks, dropped={sum(1 for track in tracks if track.dropped)}, "
        f"duration={duration_sec:.3f}s)"
    )


if __name__ == "__main__":
    main()
