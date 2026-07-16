#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cc1/aichallenge-racingkart}"
RUN_ID="${RUN_ID:-mpc-camera-rosbag-$(date +%Y%m%d-%H%M%S)}"
DURATION_SEC="${DURATION_SEC:-100}"
DOMAIN_ID="${DOMAIN_ID:-1}"
CONTROL_METHOD="${CONTROL_METHOD:-mpc}"
REALTIME_COMMENTARY="${REALTIME_COMMENTARY:-false}"
REALTIME_COMMENTARY_AFTER_CAPTURE="${REALTIME_COMMENTARY_AFTER_CAPTURE:-false}"
REALTIME_IMAGE_TOPIC="${REALTIME_IMAGE_TOPIC:-/sensing/camera/image_raw}"
REALTIME_INTERVAL_SEC="${REALTIME_INTERVAL_SEC:-3.0}"
REALTIME_PREPROCESS="${REALTIME_PREPROCESS:-lower_80x40}"
REALTIME_MIN_IMAGE_BRIGHTNESS="${REALTIME_MIN_IMAGE_BRIGHTNESS:-1.0}"
REALTIME_TEXT_MODEL="${REALTIME_TEXT_MODEL:-qwen3:8b}"
REALTIME_VISION_MODEL="${REALTIME_VISION_MODEL:-llava:7b}"
REALTIME_COMMENTARY_MODE="${REALTIME_COMMENTARY_MODE:-llm}"
REALTIME_COMMENTARY_TRIGGER="${REALTIME_COMMENTARY_TRIGGER:-interval}"
REALTIME_MIN_SPEAK_INTERVAL_SEC="${REALTIME_MIN_SPEAK_INTERVAL_SEC:-4.0}"
REALTIME_TEMPLATE_STYLE="${REALTIME_TEMPLATE_STYLE:-normal}"
REALTIME_MAX_COMMENTARY_CHARS="${REALTIME_MAX_COMMENTARY_CHARS:-45}"
REALTIME_VOICEVOX_URL="${REALTIME_VOICEVOX_URL:-}"
REALTIME_VOICEVOX_SPEAKER="${REALTIME_VOICEVOX_SPEAKER:-3}"
REALTIME_VOICEVOX_SPEED_SCALE="${REALTIME_VOICEVOX_SPEED_SCALE:-1.08}"
SCENERY_COMMENTARY="${SCENERY_COMMENTARY:-false}"
SCENERY_INTERVAL_SEC="${SCENERY_INTERVAL_SEC:-12.0}"
SCENERY_PREPROCESS="${SCENERY_PREPROCESS:-upper_320x160}"
SCENERY_MIN_IMAGE_BRIGHTNESS="${SCENERY_MIN_IMAGE_BRIGHTNESS:-2.0}"
SCENERY_VISION_MODEL="${SCENERY_VISION_MODEL:-${REALTIME_VISION_MODEL}}"
SCENERY_MAX_COMMENTARY_CHARS="${SCENERY_MAX_COMMENTARY_CHARS:-80}"
SCENERY_VOICEVOX_SPEAKER="${SCENERY_VOICEVOX_SPEAKER:-${REALTIME_VOICEVOX_SPEAKER}}"
SCENERY_VOICEVOX_SPEED_SCALE="${SCENERY_VOICEVOX_SPEED_SCALE:-1.20}"
OUT="/output/${RUN_ID}/d${DOMAIN_ID}"
HOST_OUT="${ROOT}/output/${RUN_ID}/d${DOMAIN_ID}"

cleanup() {
  docker rm -f \
    "${RUN_ID}_simulator" \
    "${RUN_ID}_reference" \
    "${RUN_ID}_awsim_adapter" \
    "${RUN_ID}_screen_recorder" \
    "${RUN_ID}_camera_viewer" \
    "${RUN_ID}_realtime_commentary" \
    "${RUN_ID}_scenery_commentary" >/dev/null 2>&1 || true
  make down >/dev/null 2>&1 || true
}

trap cleanup EXIT

cd "${ROOT}"

mkdir -p "${HOST_OUT}/capture" "${HOST_OUT}/logs"
echo "${RUN_ID}" > "${HOST_OUT}/run_id.txt"
cat > "${HOST_OUT}/run_config.txt" <<EOF
RUN_ID=${RUN_ID}
DURATION_SEC=${DURATION_SEC}
DOMAIN_ID=${DOMAIN_ID}
CONTROL_METHOD=${CONTROL_METHOD}
REALTIME_COMMENTARY=${REALTIME_COMMENTARY}
REALTIME_COMMENTARY_AFTER_CAPTURE=${REALTIME_COMMENTARY_AFTER_CAPTURE}
REALTIME_IMAGE_TOPIC=${REALTIME_IMAGE_TOPIC}
REALTIME_INTERVAL_SEC=${REALTIME_INTERVAL_SEC}
REALTIME_PREPROCESS=${REALTIME_PREPROCESS}
REALTIME_MIN_IMAGE_BRIGHTNESS=${REALTIME_MIN_IMAGE_BRIGHTNESS}
REALTIME_TEXT_MODEL=${REALTIME_TEXT_MODEL}
REALTIME_VISION_MODEL=${REALTIME_VISION_MODEL}
REALTIME_COMMENTARY_MODE=${REALTIME_COMMENTARY_MODE}
REALTIME_COMMENTARY_TRIGGER=${REALTIME_COMMENTARY_TRIGGER}
REALTIME_MIN_SPEAK_INTERVAL_SEC=${REALTIME_MIN_SPEAK_INTERVAL_SEC}
REALTIME_TEMPLATE_STYLE=${REALTIME_TEMPLATE_STYLE}
REALTIME_MAX_COMMENTARY_CHARS=${REALTIME_MAX_COMMENTARY_CHARS}
REALTIME_VOICEVOX_URL=${REALTIME_VOICEVOX_URL}
REALTIME_VOICEVOX_SPEAKER=${REALTIME_VOICEVOX_SPEAKER}
REALTIME_VOICEVOX_SPEED_SCALE=${REALTIME_VOICEVOX_SPEED_SCALE}
SCENERY_COMMENTARY=${SCENERY_COMMENTARY}
SCENERY_INTERVAL_SEC=${SCENERY_INTERVAL_SEC}
SCENERY_PREPROCESS=${SCENERY_PREPROCESS}
SCENERY_MIN_IMAGE_BRIGHTNESS=${SCENERY_MIN_IMAGE_BRIGHTNESS}
SCENERY_VISION_MODEL=${SCENERY_VISION_MODEL}
SCENERY_MAX_COMMENTARY_CHARS=${SCENERY_MAX_COMMENTARY_CHARS}
SCENERY_VOICEVOX_SPEAKER=${SCENERY_VOICEVOX_SPEAKER}
SCENERY_VOICEVOX_SPEED_SCALE=${SCENERY_VOICEVOX_SPEED_SCALE}
EOF

echo "[mpc-run] cleanup"
make down >/dev/null 2>&1 || true
docker rm -f \
  "${RUN_ID}_simulator" \
  "${RUN_ID}_reference" \
  "${RUN_ID}_awsim_adapter" \
  "${RUN_ID}_screen_recorder" \
  "${RUN_ID}_camera_viewer" \
  "${RUN_ID}_realtime_commentary" \
  "${RUN_ID}_scenery_commentary" >/dev/null 2>&1 || true
if [[ -e "${ROOT}/aichallenge/rosbag2_autoware" ]]; then
  mv "${ROOT}/aichallenge/rosbag2_autoware" "${HOST_OUT}/logs/preexisting_rosbag2_autoware" || {
    echo "[mpc-run] fallback move root-owned rosbag2_autoware"
    docker compose run --rm --user root --entrypoint bash autoware-command -lc "
      mkdir -p '/aichallenge/.trash'
      if [ -e '/aichallenge/rosbag2_autoware' ]; then
        mv '/aichallenge/rosbag2_autoware' '/aichallenge/.trash/rosbag2_autoware_root_${RUN_ID}'
      fi
    " >/dev/null 2>&1 || true
  }
fi

if [[ -f "${ROOT}/aichallenge/rosbag_autostart.log" ]]; then
  mv "${ROOT}/aichallenge/rosbag_autostart.log" "${HOST_OUT}/logs/preexisting_rosbag_autostart.log" || true
fi

echo "[mpc-run] start AWSIM"
docker compose run -d --name "${RUN_ID}_simulator" --entrypoint bash simulator -lc "
  mkdir -p '${OUT}/logs'
  export ROS_DOMAIN_ID=0
  exec /aichallenge/simulator/AWSIM/AWSIM.x86_64 \
    --start-mode count \
    --start-count-seconds 5 \
    --vehicles 1 \
    --npcs 0 \
    --boosts 2 \
    --laps unlimited \
    --timeout unlimited \
    --steer-source ackermann \
    --sound off \
    --collisions on \
    --handicap off \
    --wall-recovery on \
    --ranking off \
    --camera gpu \
    --lidar gpu \
    -screen-fullscreen 0 \
    -screen-width 960 \
    -screen-height 540 \
    > '${OUT}/logs/awsim.log' 2>&1
" >/dev/null

sleep 8

echo "[mpc-run] start reference (${CONTROL_METHOD})"
docker compose run -d --name "${RUN_ID}_reference" --entrypoint bash autoware-command -lc "
  source /opt/ros/humble/setup.bash
  source /aichallenge/workspace/install/setup.bash
  export ROS_DOMAIN_ID='${DOMAIN_ID}'
  export ROS_HOME='${OUT}/ros'
  export ROS_LOG_DIR='${OUT}/ros/log'
  mkdir -p \"\${ROS_LOG_DIR}\" '${OUT}/logs'
  exec ros2 launch aichallenge_submit_launch reference.launch.xml \
    simulation:=true \
    use_sim_time:=true \
    launch_vehicle_interface:=false \
    control_method:='${CONTROL_METHOD}' \
    > '${OUT}/logs/reference.log' 2>&1
" >/dev/null

echo "[mpc-run] start AWSIM adapter/autostart + rosbag"
docker compose run -d --name "${RUN_ID}_awsim_adapter" --entrypoint bash autoware-command -lc "
  source /opt/ros/humble/setup.bash
  source /aichallenge/workspace/install/setup.bash
  export ROS_DOMAIN_ID='${DOMAIN_ID}'
  export ROS_HOME='${OUT}/ros'
  export ROS_LOG_DIR='${OUT}/ros/log'
  mkdir -p \"\${ROS_LOG_DIR}\" '${OUT}/logs'
  exec ros2 launch \
    \$(ros2 pkg prefix aichallenge_system_launch)/share/aichallenge_system_launch/launch/mode/awsim.launch.xml \
    capture:=false \
    rosbag:=true \
    > '${OUT}/logs/awsim_adapter.log' 2>&1
" >/dev/null

echo "[mpc-run] start camera viewer"
docker compose run -d --name "${RUN_ID}_camera_viewer" -v "${ROOT}/tools:/repo_tools:ro" --entrypoint bash autoware-command -lc "
  source /opt/ros/humble/setup.bash
  source /aichallenge/workspace/install/setup.bash
  export ROS_DOMAIN_ID='${DOMAIN_ID}'
  export CAMERA_VIEW_X=0
  export CAMERA_VIEW_Y=0
  mkdir -p '${OUT}/logs'
  exec python3 /repo_tools/rl_metrics/ros_camera_viewer.py \
    --topic /sensing/camera/image_raw \
    --window-name 'MPC Camera' \
    --width 960 \
    > '${OUT}/logs/camera_viewer.log' 2>&1
" >/dev/null

if [[ "${REALTIME_COMMENTARY}" == "true" && "${REALTIME_COMMENTARY_AFTER_CAPTURE}" != "true" ]]; then
  echo "[mpc-run] start realtime VLM commentary"
  mkdir -p "${HOST_OUT}/realtime_commentary" "${HOST_OUT}/logs"
  docker compose run -d --name "${RUN_ID}_realtime_commentary" -v "${ROOT}/tools:/repo_tools:ro" --entrypoint bash autoware-command -lc "
      source /opt/ros/humble/setup.bash
      source /aichallenge/workspace/install/setup.bash
      export ROS_DOMAIN_ID='${DOMAIN_ID}'
      exec python3 /repo_tools/rl_metrics/realtime_vlm_commentary_node.py \
        --image-topic '${REALTIME_IMAGE_TOPIC}' \
        --output-dir '${OUT}/realtime_commentary' \
        --interval-sec '${REALTIME_INTERVAL_SEC}' \
        --preprocess '${REALTIME_PREPROCESS}' \
        --min-image-brightness '${REALTIME_MIN_IMAGE_BRIGHTNESS}' \
        --commentary-mode '${REALTIME_COMMENTARY_MODE}' \
        --commentary-trigger '${REALTIME_COMMENTARY_TRIGGER}' \
        --min-speak-interval-sec '${REALTIME_MIN_SPEAK_INTERVAL_SEC}' \
        --template-style '${REALTIME_TEMPLATE_STYLE}' \
        --max-commentary-chars '${REALTIME_MAX_COMMENTARY_CHARS}' \
        --vision-model '${REALTIME_VISION_MODEL}' \
        --text-model '${REALTIME_TEXT_MODEL}' \
        --voicevox-url '${REALTIME_VOICEVOX_URL}' \
        --voicevox-speaker '${REALTIME_VOICEVOX_SPEAKER}' \
        --voicevox-speed-scale '${REALTIME_VOICEVOX_SPEED_SCALE}' \
        > '${OUT}/logs/realtime_commentary.log' 2>&1
    " >/dev/null
fi

if [[ "${SCENERY_COMMENTARY}" == "true" ]]; then
  echo "[mpc-run] start scenery VLM commentary"
  mkdir -p "${HOST_OUT}/scenery_commentary" "${HOST_OUT}/logs"
  docker compose run -d --name "${RUN_ID}_scenery_commentary" -v "${ROOT}/tools:/repo_tools:ro" --entrypoint bash autoware-command -lc "
      source /opt/ros/humble/setup.bash
      source /aichallenge/workspace/install/setup.bash
      export ROS_DOMAIN_ID='${DOMAIN_ID}'
      exec python3 /repo_tools/rl_metrics/realtime_vlm_commentary_node.py \
        --image-topic '${REALTIME_IMAGE_TOPIC}' \
        --output-dir '${OUT}/scenery_commentary' \
        --interval-sec '${SCENERY_INTERVAL_SEC}' \
        --preprocess '${SCENERY_PREPROCESS}' \
        --min-image-brightness '${SCENERY_MIN_IMAGE_BRIGHTNESS}' \
        --commentary-mode template \
        --commentary-trigger interval \
        --template-style scenery \
        --max-commentary-chars '${SCENERY_MAX_COMMENTARY_CHARS}' \
        --vision-model '${SCENERY_VISION_MODEL}' \
        --text-model '${REALTIME_TEXT_MODEL}' \
        --voicevox-url '${REALTIME_VOICEVOX_URL}' \
        --voicevox-speaker '${SCENERY_VOICEVOX_SPEAKER}' \
        --voicevox-speed-scale '${SCENERY_VOICEVOX_SPEED_SCALE}' \
        > '${OUT}/logs/scenery_commentary.log' 2>&1
    " >/dev/null
fi

echo "[mpc-run] start screen recorder"
docker compose run -d --name "${RUN_ID}_screen_recorder" --entrypoint bash autoware-command -lc "
  source /opt/ros/humble/setup.bash
  source /aichallenge/workspace/install/setup.bash
  export ROS_DOMAIN_ID='${DOMAIN_ID}'
  export ROS_HOME='${OUT}/ros'
  export ROS_LOG_DIR='${OUT}/ros/log'
  mkdir -p \"\${ROS_LOG_DIR}\" '${OUT}/logs'
  exec ros2 launch \
    \$(ros2 pkg prefix aichallenge_screen_recorder)/share/aichallenge_screen_recorder/launch/screen_recorder.launch.xml \
    output_dir:='${OUT}/capture' \
    hz:=10 \
    > '${OUT}/logs/screen_recorder.log' 2>&1
" >/dev/null

echo "[mpc-run] wait windows/topics"
sleep 18

echo "[mpc-run] start capture"
date +%s.%N > "${HOST_OUT}/logs/capture_start_wall_time_sec.txt"
docker compose run --rm --no-deps --entrypoint bash autoware-command -lc "
  source /opt/ros/humble/setup.bash
  source /aichallenge/workspace/install/setup.bash
  export ROS_DOMAIN_ID='${DOMAIN_ID}'
  ros2 service call /debug/service/capture_screen std_srvs/srv/Trigger '{}'
" > "${HOST_OUT}/logs/capture_start.log" 2>&1 || true

if [[ "${REALTIME_COMMENTARY}" == "true" && "${REALTIME_COMMENTARY_AFTER_CAPTURE}" == "true" ]]; then
  echo "[mpc-run] start realtime VLM commentary after capture"
  mkdir -p "${HOST_OUT}/realtime_commentary" "${HOST_OUT}/logs"
  docker compose run -d --name "${RUN_ID}_realtime_commentary" -v "${ROOT}/tools:/repo_tools:ro" --entrypoint bash autoware-command -lc "
      source /opt/ros/humble/setup.bash
      source /aichallenge/workspace/install/setup.bash
      export ROS_DOMAIN_ID='${DOMAIN_ID}'
      exec python3 /repo_tools/rl_metrics/realtime_vlm_commentary_node.py \
        --image-topic '${REALTIME_IMAGE_TOPIC}' \
        --output-dir '${OUT}/realtime_commentary' \
        --interval-sec '${REALTIME_INTERVAL_SEC}' \
        --preprocess '${REALTIME_PREPROCESS}' \
        --min-image-brightness '${REALTIME_MIN_IMAGE_BRIGHTNESS}' \
        --commentary-mode '${REALTIME_COMMENTARY_MODE}' \
        --commentary-trigger '${REALTIME_COMMENTARY_TRIGGER}' \
        --min-speak-interval-sec '${REALTIME_MIN_SPEAK_INTERVAL_SEC}' \
        --template-style '${REALTIME_TEMPLATE_STYLE}' \
        --max-commentary-chars '${REALTIME_MAX_COMMENTARY_CHARS}' \
        --vision-model '${REALTIME_VISION_MODEL}' \
        --text-model '${REALTIME_TEXT_MODEL}' \
        --voicevox-url '${REALTIME_VOICEVOX_URL}' \
        --voicevox-speaker '${REALTIME_VOICEVOX_SPEAKER}' \
        --voicevox-speed-scale '${REALTIME_VOICEVOX_SPEED_SCALE}' \
        > '${OUT}/logs/realtime_commentary.log' 2>&1
    " >/dev/null
fi

sleep "${DURATION_SEC}"

echo "[mpc-run] stop capture"
date +%s.%N > "${HOST_OUT}/logs/capture_stop_wall_time_sec.txt"
docker compose run --rm --no-deps --entrypoint bash autoware-command -lc "
  source /opt/ros/humble/setup.bash
  source /aichallenge/workspace/install/setup.bash
  export ROS_DOMAIN_ID='${DOMAIN_ID}'
  ros2 service call /debug/service/capture_screen std_srvs/srv/Trigger '{}'
" > "${HOST_OUT}/logs/capture_stop.log" 2>&1 || true

sleep 2

echo "[mpc-run] collect status"
docker ps --format '{{.Names}} {{.Status}}' > "${HOST_OUT}/logs/docker_ps_after.txt" || true
if [[ -e "${ROOT}/aichallenge/rosbag2_autoware" ]]; then
  mkdir -p "${HOST_OUT}/rosbag2_autoware"
  cp -a "${ROOT}/aichallenge/rosbag2_autoware/." "${HOST_OUT}/rosbag2_autoware/" || true
fi
if [[ -f "${ROOT}/aichallenge/rosbag_autostart.log" ]]; then
  cp -a "${ROOT}/aichallenge/rosbag_autostart.log" "${HOST_OUT}/rosbag_autostart.log" || true
fi
find "${HOST_OUT}/capture" -maxdepth 1 -type f -name '*.mp4' -printf '%p %s bytes\n' | tee "${HOST_OUT}/logs/mp4_files.txt"
find "${HOST_OUT}" -type f \( -name '*.db3' -o -name '*.mcap' -o -name 'metadata.yaml' \) -printf '%p %s bytes\n' | tee "${HOST_OUT}/logs/rosbag_files.txt"
find "${HOST_OUT}/realtime_commentary" -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | tee "${HOST_OUT}/logs/realtime_commentary_files.txt" || true
find "${HOST_OUT}/scenery_commentary" -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | tee "${HOST_OUT}/logs/scenery_commentary_files.txt" || true

echo "[mpc-run] cleanup containers"
cleanup
trap - EXIT

echo "[mpc-run] done ${RUN_ID}"
