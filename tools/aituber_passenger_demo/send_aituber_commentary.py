#!/usr/bin/env python3
"""Send AI Challenge commentary lines to aituber-server /send_message."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_commentary(path: Path, limit: int) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for raw in file:
            if not raw.strip():
                continue
            data = json.loads(raw)
            text = (
                data.get("commentary")
                or data.get("text")
                or data.get("message")
                or data.get("speech_text")
                or ""
            )
            text = str(text).strip()
            if text:
                lines.append(text)
            if limit and len(lines) >= limit:
                break
    return lines


def post_message(endpoint: str, text: str) -> None:
    payload = json.dumps({"message": text, "text": text}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST failed: {exc.code} {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commentary-jsonl", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/send_message")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--interval-sec", type=float, default=2.5)
    parser.add_argument("--prefix", default="助手席AI: ")
    args = parser.parse_args()

    lines = load_commentary(Path(args.commentary_jsonl), args.limit)
    if not lines:
        raise SystemExit("no commentary lines found")

    for index, line in enumerate(lines):
        text = f"{args.prefix}{line}" if args.prefix else line
        post_message(args.endpoint, text)
        print(json.dumps({"index": index, "sent": text}, ensure_ascii=False), flush=True)
        if index + 1 < len(lines):
            time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
