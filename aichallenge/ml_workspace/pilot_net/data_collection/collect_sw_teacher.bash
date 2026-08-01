#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
COLLECTION_ID="${COLLECTION_ID:-sw_mpc_$(date +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-01}"
APPEND_COLLECTION="${APPEND_COLLECTION:-0}"
SPEEDS_KMH="${SPEEDS_KMH:-15 20 25}"
RECORD_SECONDS="${RECORD_SECONDS:-150}"
WARMUP_SECONDS="${WARMUP_SECONDS:-15}"
DRIVE_CHECK_SECONDS="${DRIVE_CHECK_SECONDS:-8}"
COLLECTION_DIR="${SCRIPT_DIR}/collections/${COLLECTION_ID}"
RAW_DIR="${COLLECTION_DIR}/raw"

if [[ ! "${COLLECTION_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "COLLECTION_ID contains unsupported characters: ${COLLECTION_ID}" >&2
  exit 2
fi
if [[ -e "${COLLECTION_DIR}" && "${APPEND_COLLECTION}" != "1" ]]; then
  echo "Collection already exists: ${COLLECTION_DIR}" >&2
  exit 2
fi
if [[ ! "${RUN_ID}" =~ ^[0-9][0-9]$ ]]; then
  echo "RUN_ID must contain exactly two digits: ${RUN_ID}" >&2
  exit 2
fi

mkdir -p "${RAW_DIR}"
cleanup() {
  cd "${REPO_ROOT}"
  make down >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

cd "${REPO_ROOT}"
make down >/dev/null 2>&1 || true

if [[ ! -e "${COLLECTION_DIR}/collection.yaml" ]]; then
  cat >"${COLLECTION_DIR}/collection.yaml" <<EOF
schema_version: 1
collection_id: ${COLLECTION_ID}
teacher:
  division: sw
  controller: mpc
track: citycircuit
speeds_kmh: [$(tr ' ' ',' <<<"${SPEEDS_KMH}")]
record_seconds_per_sequence: ${RECORD_SECONDS}
topics:
  image: /sensing/camera/image_raw
  control: /control/command/control_cmd
  actual_steer: /vehicle/status/steering_status
  wheel_odometry: /vehicle/status/velocity_status
notes: raw bags and extracted arrays are intentionally excluded from Git
EOF
fi

for speed in ${SPEEDS_KMH}; do
  if [[ ! "${speed}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid speed: ${speed}" >&2
    exit 2
  fi
  speed_tag="$(printf '%03d' "${speed%.*}")"
  speed_value="$(printf '%.1f' "${speed}")"
  sequence="v${speed_tag}kmh_run${RUN_ID}"
  if [[ -e "${RAW_DIR}/${sequence}" ]]; then
    echo "Sequence already exists: ${sequence}" >&2
    exit 2
  fi
  echo "Collecting ${sequence} for ${RECORD_SECONDS}s"

  # A fresh simulator per sequence avoids stale localization/control state after
  # repeated AWSIM resets and makes each sequence independently reproducible.
  make down >/dev/null 2>&1 || true
  LOG_DIR="/output/teacher-${COLLECTION_ID}-${sequence}" SIM_MODE=dev ROS_DOMAIN_ID=0 docker compose up -d simulator
  LOG_DIR="/output/teacher-${COLLECTION_ID}-${sequence}" RUN_MODE=awsim CONTROL_METHOD=mpc ROS_DOMAIN_ID=1 docker compose up -d autoware
  echo "Waiting ${WARMUP_SECONDS}s for AWSIM and MPC..."
  sleep "${WARMUP_SECONDS}"

  # MPC has per-section reference velocities which take precedence over v_max.
  # Set both the global limit and every section to obtain a real constant-speed
  # teacher profile without modifying the SW controller's checked-in config.
  docker compose exec -T autoware bash -lc "
    source /aichallenge/workspace/install/setup.bash
    ros2 param set /mpc_controller v_max ${speed_value}
    ros2 param list /mpc_controller | awk '/ref_vel\\/.+\\/ref_vel/ {print \$1}' | while read -r parameter; do
      ros2 param set /mpc_controller \"\${parameter}\" ${speed_value}
    done
    ros2 param get /mpc_controller v_max
  "

  CMD="ros2 service call /set_initial_pose std_srvs/srv/Trigger '{}'" \
    docker compose run --rm --no-deps autoware-command || true
  CMD="ros2 topic pub -1 /awsim/control_mode_request_topic std_msgs/msg/Bool '{data: true}'" \
    docker compose run --rm --no-deps autoware-command || true
  sleep "${DRIVE_CHECK_SECONDS}"
  observed_speed="$(docker compose exec -T autoware bash -lc '
    source /aichallenge/workspace/install/setup.bash
    timeout 10s ros2 topic echo --once --field longitudinal_velocity /vehicle/status/velocity_status | head -n 1
  ')"
  if ! awk -v speed="${observed_speed:-0}" 'BEGIN {if (speed < 0) speed = -speed; exit !(speed >= 0.2)}'; then
    echo "Vehicle failed pre-record drive check for ${sequence}: ${observed_speed:-missing} m/s" >&2
    exit 3
  fi
  echo "Pre-record speed check: ${observed_speed} m/s"

  docker compose exec -T autoware bash -lc "
    source /aichallenge/workspace/install/setup.bash
    timeout --signal=INT --kill-after=15s ${RECORD_SECONDS}s ros2 bag record \
      /admin/awsim/state \
      /control/command/actuation_cmd \
      /control/command/control_cmd \
      /sensing/camera/image_raw \
      /vehicle/status/steering_status \
      /vehicle/status/velocity_status \
      -o /aichallenge/ml_workspace/pilot_net/data_collection/collections/${COLLECTION_ID}/raw/${sequence} \
      -s mcap --compression-format zstd --compression-mode file
  " || status=$?
  record_status="${status:-0}"
  if [[ "${record_status}" -ne 0 && "${record_status}" -ne 124 ]]; then
    echo "rosbag recording failed for ${sequence} (status=${record_status})" >&2
    exit "${record_status}"
  fi
  unset status record_status
  make down >/dev/null 2>&1 || true
done

trap - EXIT INT TERM
cleanup
echo "Collection saved under ${COLLECTION_DIR}"
