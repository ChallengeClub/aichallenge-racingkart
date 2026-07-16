#!/usr/bin/env python3
"""Mix realtime commentary wav files according to realtime_commentary.jsonl."""

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
    requested_start: float
    play_start: float
    duration: float
    path: Path
    text: str
    index: int
    image_offset: float
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


def load_tracks(
    jsonl_path: Path,
    audio_dir: Path,
    start_mode: str,
    queue: bool,
    drop_stale_sec: float | None,
    timeline_start_wall_sec: float | None,
) -> list[Track]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"no records in {jsonl_path}")

    first_image_stamp = float(rows[0]["image_stamp_sec"])
    first_wall_time = float(rows[0].get("generation_wall_time_sec") or rows[0]["vehicle_data"]["wall_time_sec"])
    if timeline_start_wall_sec is not None:
        first_wall_time = timeline_start_wall_sec
    tracks = []
    next_available = 0.0
    for row in rows:
        idx = int(row["index"])
        path = audio_dir / f"line_{idx:04d}.wav"
        duration = wav_duration(path)
        image_offset = float(row["image_stamp_sec"]) - first_image_stamp
        generation_wall = float(row.get("generation_wall_time_sec") or row["vehicle_data"]["wall_time_sec"])
        ready_wall = float(row.get("audio_ready_wall_time_sec") or (generation_wall + float(row.get("latency_sec", {}).get("total") or 0.0)))
        generation_offset = generation_wall - first_wall_time
        total_latency = float(row.get("latency_sec", {}).get("total") or 0.0)
        ready_offset = ready_wall - first_wall_time

        if start_mode == "image_stamp":
            requested_start = image_offset
        elif start_mode == "generation_wall":
            requested_start = generation_offset
        elif start_mode == "audio_ready":
            requested_start = ready_offset
        else:
            raise ValueError(start_mode)

        queued_start = max(requested_start, next_available) if queue else requested_start
        dropped = False
        dropped_reason = ""
        if requested_start < 0.0:
            dropped = True
            dropped_reason = "before_timeline_start"
        elif drop_stale_sec is not None and (queued_start - generation_offset) > drop_stale_sec:
            dropped = True
            dropped_reason = "stale"
        play_start = queued_start
        if queue and not dropped:
            next_available = play_start + duration

        tracks.append(
            Track(
                requested_start=requested_start,
                play_start=play_start,
                duration=duration,
                path=path,
                text=str(row["commentary"]),
                index=idx,
                image_offset=image_offset,
                generation_offset=generation_offset,
                ready_offset=ready_offset,
                total_latency=total_latency,
                dropped=dropped,
                dropped_reason=dropped_reason,
            )
        )
    return tracks


def mix_tracks(tracks: list[Track], duration_sec: float, output_wav: Path) -> int:
    rate = channels = sampwidth = None
    loaded = []
    for track in tracks:
        if track.dropped:
            continue
        if track.play_start >= duration_sec:
            continue
        with wave.open(str(track.path), "rb") as wav:
            fmt = (wav.getframerate(), wav.getnchannels(), wav.getsampwidth())
            if rate is None:
                rate, channels, sampwidth = fmt
            if fmt != (rate, channels, sampwidth):
                raise RuntimeError(f"wav format mismatch: {track.path}")
            data = wav.readframes(wav.getnframes())
        loaded.append((track.play_start, data, track.text))

    if rate is None or channels is None or sampwidth is None:
        raise RuntimeError("no audio tracks inside duration")

    frames = int(duration_sec * rate)
    if sampwidth != 2:
        raise RuntimeError(f"unsupported sample width: {sampwidth}")

    mixed = array.array("h", [0]) * (frames * channels)
    for start, data, _ in loaded:
        offset = int(start * rate) * channels
        samples = array.array("h")
        samples.frombytes(data)
        size = min(len(samples), len(mixed) - offset)
        if size <= 0:
            continue
        for index in range(size):
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
                        "index": track.index,
                        "image_offset_sec": round(track.image_offset, 3),
                        "generation_offset_sec": round(track.generation_offset, 3),
                        "audio_ready_offset_sec": round(track.ready_offset, 3),
                        "requested_start_sec": round(track.requested_start, 3),
                        "play_start_sec": round(track.play_start, 3),
                        "audio_duration_sec": round(track.duration, 3),
                        "queue_delay_sec": round(track.play_start - track.requested_start, 3),
                        "total_delay_from_image_sec": round(track.play_start - track.image_offset, 3),
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
    work = video_path.parent.resolve()
    audio = audio_path.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{work}:/video:ro",
        "-v",
        f"{audio.parent}:/audio:ro",
        "-v",
        f"{output_mp4.parent.resolve()}:/out",
        ffmpeg_image,
        "-y",
        "-hide_banner",
        "-i",
        f"/video/{video_path.name}",
        "-i",
        f"/audio/{audio.name}",
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
    parser.add_argument("run_dir", type=Path, help="d1 run directory")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--start-mode", choices=["image_stamp", "generation_wall", "audio_ready"], default=None)
    parser.add_argument("--align", choices=["image_stamp", "generation_wall"], default=None, help="deprecated alias for --start-mode")
    parser.add_argument("--timeline-start-wall-sec", type=float, default=None)
    parser.add_argument("--timeline-start-wall-file", type=Path, default=None)
    parser.add_argument("--queue", action="store_true", help="delay audio until the previous line finishes")
    parser.add_argument("--drop-stale-sec", type=float, default=None, help="drop a line if play_start - image_time exceeds this")
    parser.add_argument("--output-wav", type=Path, default=None)
    parser.add_argument("--output-mp4", type=Path, default=None)
    parser.add_argument("--schedule-jsonl", type=Path, default=None)
    parser.add_argument("--ffmpeg-image", default="linuxserver/ffmpeg:latest")
    args = parser.parse_args()

    run_dir = args.run_dir
    jsonl_path = run_dir / "realtime_commentary" / "realtime_commentary.jsonl"
    audio_dir = run_dir / "realtime_commentary" / "voicevox"
    start_mode = args.start_mode or args.align or "image_stamp"
    timeline_start_wall_sec = args.timeline_start_wall_sec
    if args.timeline_start_wall_file:
        timeline_start_wall_sec = float(args.timeline_start_wall_file.read_text().strip())
    suffix = f"{start_mode}{'_queued' if args.queue else ''}"
    output_wav = args.output_wav or run_dir / "realtime_commentary" / f"commentary_mix_{suffix}.wav"
    tracks = load_tracks(
        jsonl_path,
        audio_dir,
        start_mode,
        args.queue,
        args.drop_stale_sec,
        timeline_start_wall_sec,
    )

    duration_sec = args.duration_sec
    if duration_sec is None:
        if args.video:
            duration_sec = probe_video_duration(args.ffmpeg_image, args.video)
        else:
            duration_sec = max(track.play_start + track.duration for track in tracks if not track.dropped)
    for track in tracks:
        if not track.dropped and track.play_start >= duration_sec:
            track.dropped = True
            track.dropped_reason = "after_timeline_end"

    count = mix_tracks(tracks, duration_sec, output_wav)
    dropped_count = sum(1 for track in tracks if track.dropped)
    print(
        f"wrote {output_wav} ({count} tracks, {duration_sec:.3f}s, "
        f"start_mode={start_mode}, queue={args.queue}, dropped={dropped_count})"
    )

    if args.schedule_jsonl:
        write_schedule(tracks, args.schedule_jsonl)
        print(f"wrote {args.schedule_jsonl}")

    if args.video and args.output_mp4:
        mux_video(args.ffmpeg_image, args.video, output_wav, args.output_mp4)
        print(f"wrote {args.output_mp4}")


if __name__ == "__main__":
    main()
