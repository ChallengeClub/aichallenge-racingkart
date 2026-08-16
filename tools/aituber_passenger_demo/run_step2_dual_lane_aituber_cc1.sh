#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cc1/aichallenge-racingkart}"
WORK="${WORK:-/home/cc1/aichallenge-work}"
NODE="${NODE:-/home/cc1/aichallenge-tools/node-v24.14.1-linux-x64/bin/node}"
RUN_ID="${RUN_ID:-aituber-vlm-step2-dual-lane-$(date +%Y%m%d-%H%M%S)}"
DURATION_SEC="${DURATION_SEC:-36}"
RECORD_SEC="${RECORD_SEC:-72}"
RECORD_TIMEOUT_SEC="${RECORD_TIMEOUT_SEC:-$((RECORD_SEC + 15))}"
RECORD_START_DELAY_SEC="${RECORD_START_DELAY_SEC:-0}"
PUBLIC_RECORD_SETUP_DELAY_SEC="${PUBLIC_RECORD_SETUP_DELAY_SEC:-6}"
FPS="${FPS:-15}"
DISPLAY_NAME="${DISPLAY_NAME:-:1}"
XAUTHORITY_PATH="${XAUTHORITY_PATH:-/run/user/1000/gdm/Xauthority}"
VIDEO_SIZE="${VIDEO_SIZE:-1920x1080}"
DESKTOP_RECORD_VIDEO_ENCODER="${DESKTOP_RECORD_VIDEO_ENCODER:-h264_nvenc}"
DESKTOP_RECORD_NICE="${DESKTOP_RECORD_NICE:-10}"
PUBLIC_WINDOW_LAYOUT="${PUBLIC_WINDOW_LAYOUT:-false}"
PUBLIC_LAYOUT_TIMEOUT_SEC="${PUBLIC_LAYOUT_TIMEOUT_SEC:-75}"
PUBLIC_AITUBER_GEOMETRY="${PUBLIC_AITUBER_GEOMETRY:-960,19,960,1061}"
PUBLIC_MPC_GEOMETRY="${PUBLIC_MPC_GEOMETRY:-55,25,875,460}"
PUBLIC_AWSIM_GEOMETRY="${PUBLIC_AWSIM_GEOMETRY:-55,585,875,480}"
PUBLIC_AUTOSTART_GEOMETRY="${PUBLIC_AUTOSTART_GEOMETRY:--1400,620,520,300}"
PUBLIC_CAPTION_ONLY="${PUBLIC_CAPTION_ONLY:-false}"
PUBLIC_RELAUNCH_AITUBER_CHROME="${PUBLIC_RELAUNCH_AITUBER_CHROME:-${PUBLIC_WINDOW_LAYOUT}}"
PUBLIC_AITUBER_CHROME_BIN="${PUBLIC_AITUBER_CHROME_BIN:-google-chrome}"
PUBLIC_AITUBER_CHROME_PROFILE="${PUBLIC_AITUBER_CHROME_PROFILE:-${WORK}/chrome-profile-public-${RUN_ID}}"
PUBLIC_AITUBER_URL="${PUBLIC_AITUBER_URL:-http://127.0.0.1:3028/}"
PUBLIC_AITUBER_CDP_PORT="${PUBLIC_AITUBER_CDP_PORT:-9230}"
RUN_STANDARD_OVERLAY="${RUN_STANDARD_OVERLAY:-true}"
RUN_CAMERA_VIEWER="${RUN_CAMERA_VIEWER:-${PUBLIC_WINDOW_LAYOUT}}"
RUN_CAMERA_RECORDING="${RUN_CAMERA_RECORDING:-false}"
RUN_INTERNAL_SCREEN_RECORDER="${RUN_INTERNAL_SCREEN_RECORDER:-false}"
PREROLL_TEXT="${PREROLL_TEXT:-いくよ、横で見てるね。}"
AITUBER_SELECTED_VRM_PATH="${AITUBER_SELECTED_VRM_PATH:-/vrm/TPAC_gorilla.vrm}"
AITUBER_VOICEVOX_SPEAKER="${AITUBER_VOICEVOX_SPEAKER:-23}"
AITUBER_VOICEVOX_SPEED_SCALE="${AITUBER_VOICEVOX_SPEED_SCALE:-1.6}"
BRIDGE_MIN_SEND_INTERVAL_SEC="${BRIDGE_MIN_SEND_INTERVAL_SEC:-2.8}"
BRIDGE_RECAP_MIN_SEND_INTERVAL_SEC="${BRIDGE_RECAP_MIN_SEND_INTERVAL_SEC:-7.0}"
BRIDGE_RECAP_MAX_AGE_SEC="${BRIDGE_RECAP_MAX_AGE_SEC:-5.0}"
BRIDGE_IMMEDIATE_DUPLICATE_COOLDOWN_SEC="${BRIDGE_IMMEDIATE_DUPLICATE_COOLDOWN_SEC:-20.0}"
BRIDGE_RECAP_DUPLICATE_COOLDOWN_SEC="${BRIDGE_RECAP_DUPLICATE_COOLDOWN_SEC:-18.0}"
BRIDGE_IMMEDIATE_EVENT_COOLDOWN_SEC="${BRIDGE_IMMEDIATE_EVENT_COOLDOWN_SEC:-10.0}"
BRIDGE_CRITICAL_EVENT_COOLDOWN_SEC="${BRIDGE_CRITICAL_EVENT_COOLDOWN_SEC:-3.0}"
BRIDGE_CRITICAL_DUPLICATE_COOLDOWN_SEC="${BRIDGE_CRITICAL_DUPLICATE_COOLDOWN_SEC:-4.0}"
BRIDGE_CRITICAL_MIN_SEND_INTERVAL_SEC="${BRIDGE_CRITICAL_MIN_SEND_INTERVAL_SEC:-0.0}"
BRIDGE_NON_CRITICAL_HOLD_SEC="${BRIDGE_NON_CRITICAL_HOLD_SEC:-0.8}"
BRIDGE_IMMEDIATE_MAX_QUEUED_SEC="${BRIDGE_IMMEDIATE_MAX_QUEUED_SEC:-0.4}"
BRIDGE_CRITICAL_MAX_QUEUED_SEC="${BRIDGE_CRITICAL_MAX_QUEUED_SEC:-5.0}"
BRIDGE_RECAP_AFTER_ANY_SEND_SEC="${BRIDGE_RECAP_AFTER_ANY_SEND_SEC:-3.5}"
BRIDGE_RECAP_MAX_QUEUED_SEC="${BRIDGE_RECAP_MAX_QUEUED_SEC:-0.4}"
BRIDGE_RECAP_RESERVE_INTERVAL_SEC="${BRIDGE_RECAP_RESERVE_INTERVAL_SEC:-0.0}"
BRIDGE_RECAP_RESERVE_MAX_AGE_SEC="${BRIDGE_RECAP_RESERVE_MAX_AGE_SEC:-18.0}"
BRIDGE_RECAP_RESERVE_MAX_QUEUED_SEC="${BRIDGE_RECAP_RESERVE_MAX_QUEUED_SEC:-1.5}"
BRIDGE_RECAP_BLOCK_AFTER_CRITICAL_SEC="${BRIDGE_RECAP_BLOCK_AFTER_CRITICAL_SEC:-12.0}"
BRIDGE_RECAP_BLOCK_AFTER_IMMEDIATE_SEC="${BRIDGE_RECAP_BLOCK_AFTER_IMMEDIATE_SEC:-3.0}"
BRIDGE_RECAP_MAX_CHARS="${BRIDGE_RECAP_MAX_CHARS:-42}"
REALTIME_SKIP_EVENTS="${REALTIME_SKIP_EVENTS:-}"
RECAP_ENABLED="${RECAP_ENABLED:-true}"
RECAP_START_DELAY_SEC="${RECAP_START_DELAY_SEC:-18}"
RECAP_INTERVAL_SEC="${RECAP_INTERVAL_SEC:-12.0}"
RECAP_PREPROCESS="${RECAP_PREPROCESS:-full_320x180}"
REALTIME_PREPROCESS="${REALTIME_PREPROCESS:-full_320x180}"
RECAP_VISION_MODEL="${RECAP_VISION_MODEL:-moondream}"
RECAP_TEXT_MODEL="${RECAP_TEXT_MODEL:-qwen2.5:1.5b}"
REALTIME_VISION_MODEL="${REALTIME_VISION_MODEL:-moondream}"
REALTIME_TEXT_MODEL="${REALTIME_TEXT_MODEL:-qwen2.5:1.5b}"
RECAP_OLLAMA_TIMEOUT_SEC="${RECAP_OLLAMA_TIMEOUT_SEC:-5.0}"
RECAP_GENERATION_DEADLINE_SEC="${RECAP_GENERATION_DEADLINE_SEC:-4.0}"
REALTIME_OLLAMA_TIMEOUT_SEC="${REALTIME_OLLAMA_TIMEOUT_SEC:-4.0}"
REALTIME_GENERATION_DEADLINE_SEC="${REALTIME_GENERATION_DEADLINE_SEC:-0.0}"
RECAP_TEMPLATE_STYLE="${RECAP_TEMPLATE_STYLE:-normal}"
RECAP_MAX_COMMENTARY_CHARS="${RECAP_MAX_COMMENTARY_CHARS:-42}"
PERSONA_STYLE="${PERSONA_STYLE:-loose_mascot}"
CHARACTER_PROFILE_FILE_HOST="${CHARACTER_PROFILE_FILE_HOST:-${ROOT}/tools/aituber_passenger_demo/aituber_character_profile.json}"
CHARACTER_PROFILE_FILE_CONTAINER="${CHARACTER_PROFILE_FILE_CONTAINER:-/repo_tools/aituber_passenger_demo/aituber_character_profile.json}"
DEFAULT_AMBIENT_CONTEXT="${DEFAULT_AMBIENT_CONTEXT:-urban_buildings,blue_white_barrier,open_sky}"
DEFAULT_COURSE_CONTEXT_FILE="${ROOT}/tools/rl_metrics/course_context/course_knowledge.json"
COURSE_CONTEXT_FILE="${COURSE_CONTEXT_FILE:-${DEFAULT_COURSE_CONTEXT_FILE}}"
if [[ -n "${COURSE_CONTEXT_FILE}" && ! -f "${COURSE_CONTEXT_FILE}" ]]; then
  echo "warning: COURSE_CONTEXT_FILE not found: ${COURSE_CONTEXT_FILE}; running without course context" >&2
  COURSE_CONTEXT_FILE=""
fi

HOST_OUT="${ROOT}/output/${RUN_ID}/d1"
DEMO_OUT="${WORK}/aituber-demo-20260729/${RUN_ID}"
IMMEDIATE_JSONL="${HOST_OUT}/realtime_commentary/realtime_commentary.jsonl"
RECAP_JSONL="${HOST_OUT}/vlm_recap_commentary/realtime_commentary.jsonl"
FF="${WORK}/aituber-kit/node_modules/@ffmpeg-installer/linux-x64/ffmpeg"
OUT_MP4="${DEMO_OUT}/desktop_step2_dual_lane_record.mp4"
OVERLAY_MP4="${DEMO_OUT}/desktop_step2_dual_lane_overlay.mp4"
RECAP_CONTAINER="${RUN_ID}_vlm_recap_commentary"

mkdir -p "${DEMO_OUT}/logs"

cleanup_extra() {
  docker rm -f "${RECAP_CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup_extra EXIT

if ! pactl list short sinks | awk '{print $2}' | grep -qx aituber_capture; then
  pactl load-module module-null-sink \
    sink_name=aituber_capture \
    sink_properties=device.description=aituber_capture >/dev/null
fi

pactl list short sink-inputs | awk '{print $1}' | while read -r input; do
  [ -n "${input}" ] && pactl move-sink-input "${input}" aituber_capture >/dev/null 2>&1 || true
done

if [[ "${PUBLIC_RELAUNCH_AITUBER_CHROME}" == "true" ]]; then
  ps -eo pid,args | awk '/[o]pt\/google\/chrome\/chrome .*chrome-profile-(recording|public)/ {print $1}' | while read -r pid; do
    [ -n "${pid}" ] && kill "${pid}" >/dev/null 2>&1 || true
  done
  sleep 2
  mkdir -p "${PUBLIC_AITUBER_CHROME_PROFILE}/Default"
  printf '{"translate":{"enabled":false},"intl":{"accept_languages":"ja-JP,ja"}}' \
    > "${PUBLIC_AITUBER_CHROME_PROFILE}/Default/Preferences"
  DISPLAY="${DISPLAY_NAME}" XAUTHORITY="${XAUTHORITY_PATH}" "${PUBLIC_AITUBER_CHROME_BIN}" \
    --remote-debugging-address=127.0.0.1 \
    --remote-debugging-port="${PUBLIC_AITUBER_CDP_PORT}" \
    --user-data-dir="${PUBLIC_AITUBER_CHROME_PROFILE}" \
    --no-first-run --no-default-browser-check --disable-extensions \
    --disable-background-networking --disable-component-update \
    --disable-features=ChromeWhatsNewUI,GlobalMediaControls,Translate,TranslateUI,TranslateBubbleUI \
    --disable-translate --lang=ja-JP --accept-lang=ja-JP,ja \
    --autoplay-policy=no-user-gesture-required \
    --window-position=970,0 --window-size=940,1080 \
    --app="${PUBLIC_AITUBER_URL}" \
    > "${DEMO_OUT}/logs/aituber_chrome_relaunch.log" 2>&1 &
  sleep 5
fi

"${NODE}" "${WORK}/chrome_cdp_eval.mjs" "http://127.0.0.1:${PUBLIC_AITUBER_CDP_PORT}" \
  "(() => { const speed = Number('${AITUBER_VOICEVOX_SPEED_SCALE}'); const speaker = String('${AITUBER_VOICEVOX_SPEAKER}'); const vrm = String('${AITUBER_SELECTED_VRM_PATH}'); const raw = localStorage.getItem('aitube-kit-settings'); const persisted = raw ? JSON.parse(raw) : { state: {}, version: 0 }; persisted.state = { ...(persisted.state || {}), selectedVrmPath: vrm, selectVoice: 'voicevox', voicevoxSpeaker: speaker, voicevoxSpeed: speed, voicevoxPitch: 0, voicevoxIntonation: 1, showAssistantText: false, showCharacterName: false, showQuickMenu: false, fixedCharacterPosition: false }; localStorage.setItem('aitube-kit-settings', JSON.stringify(persisted)); localStorage.setItem('selectedVrmPath', vrm); localStorage.setItem('voicevoxSpeaker', speaker); localStorage.setItem('voicevoxSpeed', String(speed)); localStorage.setItem('showAssistantText','false'); localStorage.setItem('showCharacterName','false'); localStorage.setItem('showQuickMenu','false'); localStorage.setItem('fixedCharacterPosition','false'); location.reload(); return JSON.stringify({ selectedVrmPath: persisted.state.selectedVrmPath, voicevoxSpeaker: persisted.state.voicevoxSpeaker, voicevoxSpeed: persisted.state.voicevoxSpeed, selectVoice: persisted.state.selectVoice }); })()" \
  > "${DEMO_OUT}/logs/localstorage.log" 2>&1 || true
sleep 6

for _ in 1 2 3 4 5 6 7 8; do
  "${NODE}" "${WORK}/chrome_cdp_eval.mjs" "http://127.0.0.1:${PUBLIC_AITUBER_CDP_PORT}" \
    "(() => { const closeButton = [...document.querySelectorAll('button')].find((button) => (button.innerText || '').trim() === '閉じる'); if (closeButton) { closeButton.click(); return 'closed'; } return 'missing'; })()" \
    >> "${DEMO_OUT}/logs/close_intro_dialog.log" 2>&1 || true
  sleep 0.5
done

RAW_AUDIO_PID=""

start_screen_recording() {
  (
    set +o pipefail
    parec \
      --device=aituber_capture.monitor \
      --format=s16le \
      --rate=44100 \
      --channels=2 \
      > "${DEMO_OUT}/aituber_audio.raw"
  ) 2> "${DEMO_OUT}/logs/parec_raw.log" &
  RAW_AUDIO_PID=$!

  (
    if [[ "${RECORD_START_DELAY_SEC}" != "0" ]]; then
      sleep "${RECORD_START_DELAY_SEC}"
    fi
    python3 - <<PY > "${DEMO_OUT}/logs/record_start.json"
import json, time
print(json.dumps({"record_start_wall_time_sec": time.time()}, ensure_ascii=False))
PY
    encoder="${DESKTOP_RECORD_VIDEO_ENCODER}"
    if [[ "${encoder}" != "libx264" ]] && ! "${FF}" -hide_banner -encoders 2>/dev/null | grep -q "${encoder}"; then
      echo "requested encoder ${encoder} not available; fallback to libx264" > "${DEMO_OUT}/logs/ffmpeg_encoder.log"
      encoder="libx264"
    else
      echo "selected encoder ${encoder}" > "${DEMO_OUT}/logs/ffmpeg_encoder.log"
    fi
    video_args=(-c:v libx264 -preset veryfast -pix_fmt yuv420p -r "${FPS}")
    if [[ "${encoder}" == "h264_nvenc" ]]; then
      video_args=(-c:v h264_nvenc -preset fast -pix_fmt yuv420p -r "${FPS}")
    fi
    set +o pipefail
    parec \
      --device=aituber_capture.monitor \
      --format=s16le \
      --rate=44100 \
      --channels=2 \
      2> "${DEMO_OUT}/logs/parec_screen_record.log" |
    DISPLAY="${DISPLAY_NAME}" XAUTHORITY="${XAUTHORITY_PATH}" timeout "${RECORD_TIMEOUT_SEC}s" nice -n "${DESKTOP_RECORD_NICE}" "${FF}" \
      -hide_banner -loglevel warning -y \
      -f x11grab -draw_mouse 0 -framerate "${FPS}" -video_size "${VIDEO_SIZE}" -i "${DISPLAY_NAME}.0" \
      -f s16le -ar 44100 -ac 2 -i pipe:0 \
      -t "${RECORD_SEC}" \
      "${video_args[@]}" \
      -c:a aac -b:a 160k \
      "${OUT_MP4}"
  ) > "${DEMO_OUT}/logs/ffmpeg_screen_record.log" 2>&1 &
  RECORD_PID=$!
}

LAYOUT_PID=""
if [[ "${PUBLIC_WINDOW_LAYOUT}" == "true" ]]; then
  (
    DISPLAY="${DISPLAY_NAME}" XAUTHORITY="${XAUTHORITY_PATH}" python3 "${WORK}/layout_aituberkit_public_windows.py" \
      --display "${DISPLAY_NAME}" \
      --timeout-sec "${PUBLIC_LAYOUT_TIMEOUT_SEC}" \
      --aituber="${PUBLIC_AITUBER_GEOMETRY}" \
      --mpc="${PUBLIC_MPC_GEOMETRY}" \
      --awsim="${PUBLIC_AWSIM_GEOMETRY}" \
      --autostart="${PUBLIC_AUTOSTART_GEOMETRY}"
  ) > "${DEMO_OUT}/logs/public_window_layout.log" 2>&1 &
  LAYOUT_PID=$!
fi

if [[ "${PUBLIC_WINDOW_LAYOUT}" == "true" ]]; then
  sleep "${PUBLIC_RECORD_SETUP_DELAY_SEC}"
else
  sleep 1
fi
start_screen_recording
sleep 0.5
python3 - <<PY > "${DEMO_OUT}/logs/preroll_send.json"
import json
import time
import urllib.request

text = ${PREROLL_TEXT@Q}
payload = json.dumps({
    "message": text,
    "text": text,
    "speechType": "normal",
    "source": "preroll",
    "metadata": {
        "source": "preroll",
        "lane": "preroll",
        "speechType": "normal",
        "priority": 20,
        "expireSec": 5.0,
    },
}, ensure_ascii=False).encode("utf-8")
start = time.time()
req = urllib.request.Request("http://127.0.0.1:8018/speech_event", data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=10) as res:
    response_body = res.read().decode("utf-8", errors="replace")
done = time.time()
try:
    response_json = json.loads(response_body)
except json.JSONDecodeError:
    response_json = {"raw_response": response_body}
print(json.dumps({
    "event": "preroll_sent",
    "index": "preroll",
    "lane": "preroll",
    "text": text,
    "post_start_wall_time_sec": start,
    "post_done_wall_time_sec": done,
    "aituber_event_id": response_json.get("eventId"),
    "aituber_post_response": response_json,
}, ensure_ascii=False))
PY

timeout "$((RECORD_SEC + 20))s" python3 "${WORK}/tail_dual_aituber_commentary_jsonl_measure.py" \
  --immediate-jsonl "${IMMEDIATE_JSONL}" \
  --recap-jsonl "${RECAP_JSONL}" \
  --endpoint http://127.0.0.1:8018/speech_event \
  --from-start \
  --min-send-interval-sec "${BRIDGE_MIN_SEND_INTERVAL_SEC}" \
  --recap-min-send-interval-sec "${BRIDGE_RECAP_MIN_SEND_INTERVAL_SEC}" \
  --recap-max-age-sec "${BRIDGE_RECAP_MAX_AGE_SEC}" \
  --immediate-duplicate-cooldown-sec "${BRIDGE_IMMEDIATE_DUPLICATE_COOLDOWN_SEC}" \
  --recap-duplicate-cooldown-sec "${BRIDGE_RECAP_DUPLICATE_COOLDOWN_SEC}" \
  --immediate-event-cooldown-sec "${BRIDGE_IMMEDIATE_EVENT_COOLDOWN_SEC}" \
  --critical-event-cooldown-sec "${BRIDGE_CRITICAL_EVENT_COOLDOWN_SEC}" \
  --critical-duplicate-cooldown-sec "${BRIDGE_CRITICAL_DUPLICATE_COOLDOWN_SEC}" \
  --critical-min-send-interval-sec "${BRIDGE_CRITICAL_MIN_SEND_INTERVAL_SEC}" \
  --non-critical-hold-sec "${BRIDGE_NON_CRITICAL_HOLD_SEC}" \
  --immediate-max-queued-sec "${BRIDGE_IMMEDIATE_MAX_QUEUED_SEC}" \
  --critical-max-queued-sec "${BRIDGE_CRITICAL_MAX_QUEUED_SEC}" \
  --recap-after-any-send-sec "${BRIDGE_RECAP_AFTER_ANY_SEND_SEC}" \
  --recap-max-queued-sec "${BRIDGE_RECAP_MAX_QUEUED_SEC}" \
  --recap-reserve-interval-sec "${BRIDGE_RECAP_RESERVE_INTERVAL_SEC}" \
  --recap-reserve-max-age-sec "${BRIDGE_RECAP_RESERVE_MAX_AGE_SEC}" \
  --recap-reserve-max-queued-sec "${BRIDGE_RECAP_RESERVE_MAX_QUEUED_SEC}" \
  --recap-block-after-critical-sec "${BRIDGE_RECAP_BLOCK_AFTER_CRITICAL_SEC}" \
  --recap-block-after-immediate-sec "${BRIDGE_RECAP_BLOCK_AFTER_IMMEDIATE_SEC}" \
  --recap-max-chars "${BRIDGE_RECAP_MAX_CHARS}" \
  --recap-prefix "" \
  --idle-timeout-sec 0 \
  > "${DEMO_OUT}/logs/dual_bridge_measure.jsonl" 2>&1 &
BRIDGE_PID=$!

set +e
cd "${ROOT}"
RUN_ID="${RUN_ID}" DURATION_SEC="${DURATION_SEC}" DOMAIN_ID=1 CONTROL_METHOD=mpc \
  REALTIME_COMMENTARY=true REALTIME_COMMENTARY_AFTER_CAPTURE=false \
  CAMERA_VIEWER="${RUN_CAMERA_VIEWER}" CAMERA_RECORDING="${RUN_CAMERA_RECORDING}" CAMERA_RECORD_FPS=10 CAMERA_VIEW_WIDTH=870 AWSIM_SCREEN_WIDTH=875 AWSIM_SCREEN_HEIGHT=480 AWSIM_SCREEN_X=55 AWSIM_SCREEN_Y=585 \
  REALTIME_COMMENTARY_MODE=template REALTIME_TEMPLATE_STYLE=event_fast \
  REALTIME_PERSONA_STYLE="${PERSONA_STYLE}" \
  REALTIME_CHARACTER_PROFILE_FILE="${CHARACTER_PROFILE_FILE_CONTAINER}" \
  REALTIME_COURSE_CONTEXT="${COURSE_CONTEXT_FILE}" \
  REALTIME_SKIP_EVENTS="${REALTIME_SKIP_EVENTS}" \
  REALTIME_COMMENTARY_TRIGGER=event REALTIME_INTERVAL_SEC=0.5 REALTIME_MIN_SPEAK_INTERVAL_SEC=1.0 \
  REALTIME_PREPROCESS="${REALTIME_PREPROCESS}" REALTIME_MAX_COMMENTARY_CHARS=34 \
  REALTIME_VISION_MODEL="${REALTIME_VISION_MODEL}" REALTIME_TEXT_MODEL="${REALTIME_TEXT_MODEL}" \
  REALTIME_OLLAMA_TIMEOUT_SEC="${REALTIME_OLLAMA_TIMEOUT_SEC}" REALTIME_GENERATION_DEADLINE_SEC="${REALTIME_GENERATION_DEADLINE_SEC}" \
  REALTIME_VOICEVOX_URL= REALTIME_VOICEVOX_SPEAKER="${AITUBER_VOICEVOX_SPEAKER}" REALTIME_VOICEVOX_SPEED_SCALE="${AITUBER_VOICEVOX_SPEED_SCALE}" \
  SCREEN_RECORDING="${RUN_INTERNAL_SCREEN_RECORDER}" \
  DISPLAY="${DISPLAY_NAME}" XAUTHORITY="${XAUTHORITY_PATH}" \
  tools/rl_metrics/run_mpc_camera_rosbag_capture.sh \
  > "${DEMO_OUT}/logs/run.log" 2>&1 &
RUN_PID=$!
set -e

mkdir -p "${HOST_OUT}/vlm_recap_commentary"
sleep "${RECAP_START_DELAY_SEC}"

if [[ "${RECAP_ENABLED}" == "true" ]]; then
docker compose run -d --name "${RECAP_CONTAINER}" -v "${ROOT}/tools:/repo_tools:ro" --entrypoint bash autoware-command -lc "
  source /opt/ros/humble/setup.bash
  source /aichallenge/workspace/install/setup.bash
  export ROS_DOMAIN_ID='1'
  exec python3 /repo_tools/rl_metrics/realtime_vlm_commentary_node.py \
    --image-topic '/sensing/camera/image_raw' \
    --output-dir '/output/${RUN_ID}/d1/vlm_recap_commentary' \
    --interval-sec '${RECAP_INTERVAL_SEC}' \
    --preprocess '${RECAP_PREPROCESS}' \
    --min-image-brightness '1.0' \
    --commentary-mode 'llm' \
    --commentary-trigger 'interval' \
    --min-speak-interval-sec '${RECAP_INTERVAL_SEC}' \
    --template-style '${RECAP_TEMPLATE_STYLE}' \
    --persona-style '${PERSONA_STYLE}' \
    --character-profile-file '${CHARACTER_PROFILE_FILE_CONTAINER}' \
    --default-ambient-context '${DEFAULT_AMBIENT_CONTEXT}' \
    --course-context '${COURSE_CONTEXT_FILE}' \
    --max-commentary-chars '${RECAP_MAX_COMMENTARY_CHARS}' \
    --vision-model '${RECAP_VISION_MODEL}' \
    --text-model '${RECAP_TEXT_MODEL}' \
    --ollama-timeout-sec '${RECAP_OLLAMA_TIMEOUT_SEC}' \
    --generation-deadline-sec '${RECAP_GENERATION_DEADLINE_SEC}' \
    --voicevox-url '' \
    --voicevox-speaker '${AITUBER_VOICEVOX_SPEAKER}' \
    --voicevox-speed-scale '${AITUBER_VOICEVOX_SPEED_SCALE}' \
    > '/output/${RUN_ID}/d1/logs/vlm_recap_commentary.log' 2>&1
" >/dev/null
else
  echo "RECAP_ENABLED=false; skip recap container" > "${DEMO_OUT}/logs/vlm_recap_disabled.log"
fi

set +e
wait "${RUN_PID}"
RUN_RC=$?
echo "${RUN_RC}" > "${DEMO_OUT}/logs/run_rc.txt"

wait "${RECORD_PID}"; echo $? > "${DEMO_OUT}/logs/record_rc.txt"
if [[ -n "${LAYOUT_PID}" ]]; then
  wait "${LAYOUT_PID}" >/dev/null 2>&1 || true
fi
wait "${BRIDGE_PID}"; echo $? > "${DEMO_OUT}/logs/bridge_rc.txt"
if [[ -n "${RAW_AUDIO_PID}" ]]; then
  kill "${RAW_AUDIO_PID}" >/dev/null 2>&1 || true
  wait "${RAW_AUDIO_PID}" >/dev/null 2>&1 || true
fi
set -e

python3 - <<PY > "${DEMO_OUT}/logs/step2_dual_lane_analysis.json"
import json
import math
import pathlib
import statistics
import struct

rate = 44100
channels = 2
bytes_per_sample = 2
threshold = 900
base = pathlib.Path("${DEMO_OUT}")
record_start = json.loads((base / "logs" / "record_start.json").read_text())["record_start_wall_time_sec"]
raw = base / "aituber_audio.raw"

def summarize(values):
    return {"count": len(values), "min": min(values) if values else None, "median": statistics.median(values) if values else None, "max": max(values) if values else None}

def read_json_lines(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def audio_segments(path):
    win_frames = int(rate * 0.02)
    win_values = win_frames * channels
    segments = []
    active = False
    seg_start = None
    last_active = None
    silence_gap = 0.35
    value_index = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(win_values * bytes_per_sample)
            if not chunk:
                break
            n = len(chunk) // bytes_per_sample
            vals = struct.unpack("<%dh" % n, chunk)
            rms = math.sqrt(sum(v * v for v in vals) / n) if n else 0.0
            t = value_index / (rate * channels)
            if rms >= threshold:
                if not active:
                    active = True
                    seg_start = t
                last_active = t + 0.02
            elif active and last_active is not None and t - last_active >= silence_gap:
                segments.append([seg_start, last_active])
                active = False
                seg_start = None
                last_active = None
            value_index += n
    if active:
        segments.append([seg_start, last_active])
    merged = []
    for s, e in segments:
        if not merged or s - merged[-1][1] > 0.5:
            merged.append([s, e])
        else:
            merged[-1][1] = e
    return merged

bridge_events = [r for r in read_json_lines(base / "logs" / "dual_bridge_measure.jsonl") if r.get("event") == "sent"]
preroll = json.loads((base / "logs" / "preroll_send.json").read_text())
events = [{
    "event": "sent",
    "lane": "preroll",
    "index": "preroll",
    "text": preroll["text"],
    "aituber_event_id": preroll.get("aituber_event_id"),
    "post_done_wall_time_sec": preroll["post_done_wall_time_sec"],
    "audio_ready_wall_time_sec": preroll["post_done_wall_time_sec"],
}] + bridge_events
segments = audio_segments(raw)
matches = []
used = set()
for ev in events:
    post_done = ev.get("post_done_wall_time_sec")
    chosen = None
    for i, (s, e) in enumerate(segments):
        onset_wall = record_start + s
        if i in used:
            continue
        if post_done is not None and onset_wall >= post_done - 0.35:
            chosen = (i, onset_wall, record_start + e, e - s, s, e)
            break
    if chosen:
        used.add(chosen[0])
    generation_wall = ev.get("generation_wall_time_sec")
    matches.append({
        "lane": ev.get("lane"),
        "index": ev.get("index"),
        "text": ev.get("text"),
        "aituber_event_id": ev.get("aituber_event_id"),
        "image_stamp_sec": ev.get("image_stamp_sec"),
        "generation_wall_time_sec": generation_wall,
        "post_done_wall_time_sec": post_done,
        "speech_start_offset_sec": chosen[4] if chosen else None,
        "speech_duration_sec": chosen[3] if chosen else None,
        "post_done_to_speech_start_sec": chosen[1] - post_done if chosen and post_done else None,
        "generation_to_speech_start_sec": chosen[1] - generation_wall if chosen and generation_wall else None,
    })

by_lane = {}
for m in matches:
    by_lane.setdefault(m.get("lane"), []).append(m)

payload = {
    "run": "${RUN_ID}",
    "first_speech_offset_sec": segments[0][0] if segments else None,
    "send_count": len(bridge_events),
    "drop_count": len([r for r in read_json_lines(base / "logs" / "dual_bridge_measure.jsonl") if r.get("event") == "drop"]),
    "sent_by_lane": {lane: len(rows) for lane, rows in by_lane.items()},
    "summary_by_lane": {
        lane: {
            "post_done_to_speech_start_sec": summarize([m["post_done_to_speech_start_sec"] for m in rows if m["post_done_to_speech_start_sec"] is not None]),
            "generation_to_speech_start_sec": summarize([m["generation_to_speech_start_sec"] for m in rows if m["generation_to_speech_start_sec"] is not None]),
        }
        for lane, rows in by_lane.items()
    },
    "matches": matches,
}
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

DEMO_OUT="${DEMO_OUT}" PUBLIC_CAPTION_ONLY="${PUBLIC_CAPTION_ONLY}" python3 - <<'PY' > "${DEMO_OUT}/logs/step2_overlay.ass"
import json
import os
import pathlib

base = pathlib.Path(os.environ["DEMO_OUT"])
data = json.loads((base / "logs" / "step2_dual_lane_analysis.json").read_text())
public_caption = os.environ.get("PUBLIC_CAPTION_ONLY", "").lower() == "true"

def esc(text):
    return str(text).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")

def ts(sec):
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h}:{m:02d}:{s:05.2f}"

lines = [
    "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1920", "PlayResY: 1080", "",
    "[V4+ Styles]",
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
    (
        "Style: info,Arial,54,&H00FFFFFF,&H000000FF,&H00111111,&H99000000,0,0,0,0,100,100,0,0,3,2,0,2,80,80,54,1"
        if public_caption
        else "Style: info,Arial,32,&H00FFFFFF,&H000000FF,&H00202020,&H99000000,0,0,0,0,100,100,0,0,3,2,0,7,34,34,42,1"
    ), "",
    "[Events]",
    "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
]
first = data.get("first_speech_offset_sec")
if first is not None and not public_caption:
    lines.append(f"Dialogue: 0,0:00:00.00,0:00:05.00,info,,0,0,0,,first speech {first:.2f}s | dual lane")
for m in data.get("matches", []):
    start = m.get("speech_start_offset_sec")
    if start is None:
        continue
    dur = m.get("speech_duration_sec") or 2.5
    lane = m.get("lane")
    delay = m.get("generation_to_speech_start_sec")
    delay_text = f"gen->speech {delay:.2f}s" if delay is not None else "preroll"
    label = esc(m.get("text", "")) if public_caption else f"{lane} {m.get('index')}: {delay_text} | {esc(m.get('text', ''))}"
    lines.append(f"Dialogue: 0,{ts(start)},{ts(start + min(max(dur, 2.0), 5.0))},info,,0,0,0,,{label}")
(base / "logs" / "step2_overlay.ass").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

if [[ "${RUN_STANDARD_OVERLAY}" == "true" ]] && [ -x "${FF}" ] && [ -f "${OUT_MP4}" ]; then
  "${FF}" -hide_banner -loglevel error -y \
    -i "${OUT_MP4}" \
    -vf "ass=${DEMO_OUT}/logs/step2_overlay.ass" \
    -c:v libx264 -preset veryfast -pix_fmt yuv420p \
    -c:a copy \
    "${OVERLAY_MP4}" || true
  "${FF}" -hide_banner -loglevel error -y -ss 4 -i "${OVERLAY_MP4}" -frames:v 1 "${DEMO_OUT}/step2_frame_4s.jpg" || true
  "${FF}" -hide_banner -loglevel error -y -ss 24 -i "${OVERLAY_MP4}" -frames:v 1 "${DEMO_OUT}/step2_frame_24s.jpg" || true
  "${FF}" -hide_banner -i "${OVERLAY_MP4}" -af volumedetect -f null - > "${DEMO_OUT}/logs/volumedetect.log" 2>&1 || true
fi

{
  echo "RUN=${RUN_ID}"
  echo "RUN_RC=${RUN_RC}"
  echo "video=${OVERLAY_MP4}"
  echo "immediate=${IMMEDIATE_JSONL}"
  echo "recap=${RECAP_JSONL}"
  echo "record_rc $(cat "${DEMO_OUT}/logs/record_rc.txt" 2>/dev/null || true)"
  echo "bridge_rc $(cat "${DEMO_OUT}/logs/bridge_rc.txt" 2>/dev/null || true)"
  echo "analysis"
  cat "${DEMO_OUT}/logs/step2_dual_lane_analysis.json"
  echo "volume"
  tail -n 8 "${DEMO_OUT}/logs/volumedetect.log" 2>/dev/null || true
} | tee "${DEMO_OUT}/summary_step2.txt"

cleanup_extra
trap - EXIT
exit "${RUN_RC}"
