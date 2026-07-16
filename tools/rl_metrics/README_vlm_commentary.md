# VLM Driving Commentary Tools

This directory contains experimental tools for generating VLM-based driving commentary from AI Challenge / AWSIM camera videos and ROS 2 bags.

The tools are intended for research, visualization, and team discussion. They are not required by the official evaluation runtime.

## References

- Current write-up: https://qiita.com/kiwsdiv/items/94ff578a486f022119f0
- Sample driving video: https://www.youtube.com/watch?v=o4JYSiev4UI

## What This Provides

- Offline commentary generation from a camera video and a synchronized rosbag.
- Semi-realtime ROS 2 node that subscribes to camera and vehicle-state topics.
- VOICEVOX narration synthesis.
- Video/audio mixing with timing and delay overlays.
- Camera preprocessing experiments for local VLM input.

The current design assumes a local VLM stack such as Ollama with `llava:7b` for vision and `qwen3:8b` for text generation. Template-only modes are also available for lower latency.

## Main Scripts

| Script | Purpose |
| --- | --- |
| `synced_mcap_vlm_commentary.py` | Generate commentary from a recorded camera video and rosbag/mcap vehicle data. |
| `realtime_vlm_commentary_node.py` | ROS 2 node for semi-realtime VLM/template commentary. |
| `run_mpc_camera_rosbag_capture.sh` | Helper runner for MPC camera capture, rosbag recording, and optional realtime commentary. |
| `make_realtime_commentary_mix.py` | Mix realtime commentary WAV files into a capture video and write delay overlays. |
| `make_dual_realtime_commentary_mix.py` | Mix primary driving commentary and secondary scenery commentary. |
| `make_voicevox_commentary_audio.py` | Convert commentary JSONL into a scheduled VOICEVOX narration WAV. |
| `compare_camera_preprocessing_vlm.py` | Compare camera preprocessing variants with local VLM commentary. |
| `benchmark_vlm_preprocessing_tags.py` | Benchmark local VLM tag extraction across preprocessing variants. |
| `sensor_grounded_vlm_commentary.py` | Generate sensor-grounded VLM commentary from sampled frames and vehicle data. |
| `ros_camera_viewer.py` | Simple ROS 2 camera viewer for monitoring. |

## Requirements

Run inside the AI Challenge development environment where ROS 2 Humble and the AI Challenge workspace are available.

Typical additional dependencies:

- Python packages: `opencv-python`, `numpy`
- ROS Python packages: `rclpy`, `cv_bridge`, `rosbag2_py`, `rosidl_runtime_py`
- Local VLM server: Ollama
- Optional TTS server: VOICEVOX
- Optional video mixing: ffmpeg or the `linuxserver/ffmpeg:latest` container image

Example Ollama models:

```bash
ollama pull llava:7b
ollama pull qwen3:8b
```

VOICEVOX is optional. Without VOICEVOX, the tools still write text JSONL output.

## Offline Commentary From Recorded Video

Use this mode first when sharing with others. It is easier to reproduce than the realtime node.

```bash
python3 tools/rl_metrics/synced_mcap_vlm_commentary.py \
  --video output/<run_id>/d1/capture/<capture>.mp4 \
  --bag output/<run_id>/d1/rosbag2_autoware \
  --output-dir output/vlm-commentary/<run_id> \
  --times 5,15,25,35,45,55 \
  --preprocess lower_half_160x80 \
  --vision-model llava:7b \
  --text-model qwen3:8b
```

Outputs include:

- `commentary.jsonl`
- synchronized vehicle data
- sampled/preprocessed frames
- generated commentary text

If video start and bag start are offset, set:

```bash
--bag-video-offset-sec <seconds>
```

## Scheduled VOICEVOX Narration

After generating `commentary.jsonl`, create a narration WAV:

```bash
python3 tools/rl_metrics/make_voicevox_commentary_audio.py \
  --commentary-jsonl output/vlm-commentary/<run_id>/commentary.jsonl \
  --output-wav output/vlm-commentary/<run_id>/commentary.wav \
  --manifest output/vlm-commentary/<run_id>/voicevox_manifest.json \
  --voicevox-url http://127.0.0.1:50021 \
  --speaker 3
```

## Semi-Realtime Commentary Node

`realtime_vlm_commentary_node.py` subscribes to camera and vehicle-state topics and writes commentary records as JSONL. It can call a VLM, use template-only output, and optionally synthesize VOICEVOX audio.

Basic example:

```bash
python3 tools/rl_metrics/realtime_vlm_commentary_node.py \
  --image-topic /sensing/camera/image_raw \
  --odom-topic /localization/kinematic_state \
  --control-topic /control/command/control_cmd \
  --accel-topic /localization/acceleration \
  --output-dir /tmp/realtime_vlm_commentary \
  --interval-sec 3.0 \
  --preprocess lower_80x40 \
  --commentary-mode template \
  --template-style event_fast
```

VLM + text model example:

```bash
python3 tools/rl_metrics/realtime_vlm_commentary_node.py \
  --output-dir /tmp/realtime_vlm_commentary \
  --interval-sec 3.0 \
  --preprocess lower_80x40 \
  --commentary-mode llm \
  --vision-model llava:7b \
  --text-model qwen3:8b
```

VOICEVOX example:

```bash
python3 tools/rl_metrics/realtime_vlm_commentary_node.py \
  --output-dir /tmp/realtime_vlm_commentary \
  --commentary-mode template \
  --template-style event_fast \
  --voicevox-url http://127.0.0.1:50021 \
  --voicevox-speaker 3 \
  --voicevox-speed-scale 1.30
```

Low-latency recommendation:

- `--commentary-mode template`
- `--template-style event_fast`
- `--commentary-trigger event`
- `--min-speak-interval-sec 3.0`
- VOICEVOX speed scale around `1.30`

This avoids waiting for VLM inference on every utterance.

## Capture Runner With Optional Commentary

`run_mpc_camera_rosbag_capture.sh` can record camera video and rosbag, and optionally launch the realtime commentary node.

Example:

```bash
RUN_ID=mpc-vlm-commentary-demo \
REALTIME_COMMENTARY=true \
REALTIME_COMMENTARY_MODE=template \
REALTIME_COMMENTARY_TRIGGER=event \
REALTIME_TEMPLATE_STYLE=event_fast \
REALTIME_VOICEVOX_URL=http://127.0.0.1:50021 \
REALTIME_VOICEVOX_SPEED_SCALE=1.30 \
tools/rl_metrics/run_mpc_camera_rosbag_capture.sh
```

Important environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `REALTIME_COMMENTARY` | `false` | Enable realtime commentary node. |
| `REALTIME_IMAGE_TOPIC` | `/sensing/camera/image_raw` | Camera topic. |
| `REALTIME_INTERVAL_SEC` | `3.0` | Commentary sampling interval. |
| `REALTIME_PREPROCESS` | `lower_80x40` | Camera preprocessing for VLM input. |
| `REALTIME_COMMENTARY_MODE` | `llm` | `llm` or `template`. |
| `REALTIME_COMMENTARY_TRIGGER` | `interval` | `interval` or `event`. |
| `REALTIME_TEMPLATE_STYLE` | `normal` | `normal`, `short`, `event`, `event_fast`, etc. |
| `REALTIME_VISION_MODEL` | `llava:7b` | Ollama vision model. |
| `REALTIME_TEXT_MODEL` | `qwen3:8b` | Ollama text model. |
| `REALTIME_VOICEVOX_URL` | empty | If set, synthesize WAV files. |
| `REALTIME_VOICEVOX_SPEAKER` | `3` | VOICEVOX speaker id. |
| `REALTIME_VOICEVOX_SPEED_SCALE` | `1.08` | Speech speed. |

## Mixing Realtime Commentary Into Video

After a realtime run, mix generated WAV files into the capture video:

```bash
python3 tools/rl_metrics/make_realtime_commentary_mix.py \
  output/<run_id>/d1 \
  --video output/<run_id>/d1/capture/<capture>.mp4 \
  --timeline-start-wall-file output/<run_id>/d1/logs/capture_start_wall_time_sec.txt \
  --start-mode audio_ready \
  --queue \
  --drop-stale-sec 4.0 \
  --output-mp4 output/<run_id>/d1/capture/<capture>_vlm_commentary.mp4 \
  --schedule-jsonl output/<run_id>/d1/realtime_commentary_schedule.jsonl
```

Use `--timeline-start-wall-file` when available. It aligns audio playback to the screen-capture wall-clock timeline instead of assuming the first commentary record starts at video time zero.

## Output Files

Typical realtime output directory:

```text
realtime_commentary.jsonl
realtime_commentary.md
audio/
  line_000.wav
  line_001.wav
```

Each JSONL record may include:

- image timestamp
- vehicle speed
- control command
- generated text
- VLM/text/TTS latency
- audio file path
- wall-clock timing fields

## Current Limitations

- The realtime node is experimental and intended for visualization/research.
- VLM latency depends heavily on model size, GPU/CPU, and image preprocessing.
- Template modes are more stable for realtime timing than LLM-generated Japanese prose.
- VOICEVOX audio can queue up if utterances are too long or too frequent.
- Camera preprocessing is tuned for the AI Challenge camera view and may need adjustment for other views.
- The scripts assume local access to the AI Challenge ROS 2 topics and recorded output layout.
- This branch does not modify the official evaluation behavior.

## Suggested Sharing Flow

1. Start with offline `synced_mcap_vlm_commentary.py` on a known capture and bag.
2. Share generated `commentary.jsonl` and a sample video with narration.
3. Then try `realtime_vlm_commentary_node.py` in template/event mode.
4. Use VLM inference as an optional enhancement, not as the first dependency.

