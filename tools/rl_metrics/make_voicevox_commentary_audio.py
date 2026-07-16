#!/usr/bin/env python3
"""Create a scheduled VOICEVOX narration WAV from commentary.jsonl."""

from __future__ import annotations

import argparse
import array
import json
from pathlib import Path
import urllib.parse
import urllib.request
import wave


def synthesize_line(base_url: str, speaker: int, text: str, output_path: Path) -> None:
    query_url = f"{base_url.rstrip('/')}/audio_query?" + urllib.parse.urlencode(
        {"text": text, "speaker": speaker}
    )
    req = urllib.request.Request(query_url, method="POST")
    with urllib.request.urlopen(req, timeout=30) as response:
        query = json.loads(response.read().decode("utf-8"))

    query["speedScale"] = 1.08
    query["prePhonemeLength"] = 0.05
    query["postPhonemeLength"] = 0.08

    synth_url = f"{base_url.rstrip('/')}/synthesis?" + urllib.parse.urlencode(
        {"speaker": speaker}
    )
    req = urllib.request.Request(
        synth_url,
        data=json.dumps(query).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        output_path.write_bytes(response.read())


def read_mono_wav(path: Path) -> tuple[int, array.array]:
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        pcm = wav.readframes(wav.getnframes())
    if sample_width != 2:
        raise RuntimeError(f"unsupported sample width: {sample_width}")
    samples = array.array("h")
    samples.frombytes(pcm)
    if channels > 1:
        mono = array.array("h")
        for i in range(0, len(samples), channels):
            mono.append(int(sum(samples[i : i + channels]) / channels))
        samples = mono
    return rate, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commentary-jsonl", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--voicevox-url", default="http://127.0.0.1:50021")
    parser.add_argument("--speaker", type=int, default=3)
    parser.add_argument("--duration-sec", type=float, default=38.5)
    parser.add_argument("--padding-sec", type=float, default=0.2)
    args = parser.parse_args()

    commentary_jsonl = Path(args.commentary_jsonl)
    output_wav = Path(args.output_wav)
    manifest_path = Path(args.manifest)
    wav_dir = output_wav.parent / "voicevox"
    wav_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with commentary_jsonl.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    segments = []
    for record in records:
        idx = int(record.get("index", 0))
        text = record["commentary"]
        wav_path = wav_dir / f"line_{idx:02d}.wav"
        synthesize_line(args.voicevox_url, args.speaker, text, wav_path)
        rate, samples = read_mono_wav(wav_path)
        segments.append((float(record["time_sec"]), rate, samples, text, wav_path))

    if not segments:
        raise RuntimeError("no commentary records")

    sample_rate = segments[0][1]
    total = array.array("h", [0]) * int(args.duration_sec * sample_rate)
    for start_sec, rate, samples, text, wav_path in segments:
        if rate != sample_rate:
            raise RuntimeError("mixed sample rates are not supported")
        start = int((start_sec + args.padding_sec) * sample_rate)
        for i, value in enumerate(samples):
            j = start + i
            if j >= len(total):
                break
            mixed = total[j] + value
            total[j] = max(-32768, min(32767, mixed))

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(total.tobytes())

    manifest_path.write_text(
        json.dumps(
            {
                "speaker": args.speaker,
                "sample_rate": sample_rate,
                "duration_sec": len(total) / sample_rate,
                "segments": [
                    {"time_sec": s, "text": t, "wav": str(p)}
                    for s, _, _, t, p in segments
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_wav)


if __name__ == "__main__":
    main()
