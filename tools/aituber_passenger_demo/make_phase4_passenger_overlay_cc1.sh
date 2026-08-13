#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-aituber-passenger-demo-20260727}"
ROOT="${ROOT:-/home/cc1/aichallenge-racingkart}"
OUT_DIR="${OUT_DIR:-${ROOT}/output/${RUN_ID}}"
INPUT_VIDEO="${INPUT_VIDEO:-${ROOT}/output/mpc-realtime-commentary-20260713-0035-template-voice/d1/capture/cap-20260713-003716_voicevox_realtime_queued_delay_overlay.mp4}"
OUTPUT_VIDEO="${OUTPUT_VIDEO:-${OUT_DIR}/aituber_passenger_phase4_demo_v2_h264.mp4}"
DURATION_SEC="${DURATION_SEC:-32}"
FFMPEG_IMAGE="${FFMPEG_IMAGE:-linuxserver/ffmpeg:latest}"
FONT_NAME="${FONT_NAME:-Noto Sans CJK JP}"

mkdir -p "${OUT_DIR}"

ASS_FILE="${OUT_DIR}/passenger_overlay.ass"
cat >"${ASS_FILE}" <<EOF
[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Bubble,${FONT_NAME},28,&H00111827,&H00111827,&H00FFFFFF,&HEEFFFFFF,1,0,0,0,100,100,0,0,3,1,0,7,996,52,390,1
Style: Small,${FONT_NAME},21,&H00DDE7F0,&H00DDE7F0,&H00111827,&H99000000,0,0,0,0,100,100,0,0,1,2,0,7,972,40,92,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 5,0:00:00.00,0:00:32.00,Bubble,,0,0,0,,Dory\\N助手席AI
Dialogue: 5,0:00:00.00,0:00:32.00,Small,,0,0,0,,visual mock / audio: source VOICEVOX
EOF

docker run --rm \
  -v "${OUT_DIR}:/work" \
  -v "${INPUT_VIDEO}:/input.mp4:ro" \
  -v /usr/share/fonts:/usr/share/fonts:ro \
  -v /etc/fonts:/etc/fonts:ro \
  "${FFMPEG_IMAGE}" \
  -hide_banner -loglevel warning -y \
  -t "${DURATION_SEC}" \
  -i /input.mp4 \
  -filter_complex "[0:v]scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,setsar=1,drawbox=x=950:y=72:w=300:h=600:color=0x0F172A@0.70:t=fill,drawbox=x=980:y=118:w=238:h=238:color=0xF8FAFC@0.95:t=fill,drawbox=x=1018:y=176:w=38:h=38:color=0x111827@1:t=fill,drawbox=x=1142:y=176:w=38:h=38:color=0x111827@1:t=fill,drawbox=x=1062:y=254:w=74:h=16:color=0xEF4444@1:t=fill,drawbox=x=995:y=382:w=210:h=136:color=0xFFFFFF@0.92:t=fill,drawbox=x=995:y=382:w=210:h=136:color=0x38BDF8@1:t=4,ass=/work/passenger_overlay.ass[v]" \
  -map "[v]" -map 0:a? \
  -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 23 \
  -c:a aac -b:a 160k -movflags +faststart \
  /work/$(basename "${OUTPUT_VIDEO}")

docker run --rm \
  -v "${OUT_DIR}:/work" \
  "${FFMPEG_IMAGE}" \
  -hide_banner -loglevel warning -y \
  -ss 00:00:08 -i "/work/$(basename "${OUTPUT_VIDEO}")" \
  -frames:v 1 -update 1 /work/phase4_v2_frame_8s.jpg

cat >"${OUT_DIR}/README.md" <<EOF
# ${RUN_ID}

- input: ${INPUT_VIDEO}
- output: ${OUTPUT_VIDEO}
- duration_sec: ${DURATION_SEC}
- generated_at: $(date -Is)

This is a Phase 4 MVP video: existing AI Challenge commentary video plus passenger-character style overlay.
The previous v1 used hard-coded top subtitles that did not match the source audio; v2 removes those subtitles.
The AITuberKit live browser capture can replace this overlay after the external-linkage smoke is complete.
EOF

printf '%s\n' "${OUTPUT_VIDEO}"
