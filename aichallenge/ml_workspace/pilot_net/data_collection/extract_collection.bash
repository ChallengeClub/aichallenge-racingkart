#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
COLLECTION_DIR="${1:-}"
if [[ -z "${COLLECTION_DIR}" || ! -d "${COLLECTION_DIR}/raw" ]]; then
  echo "Usage: $0 <collection-directory>" >&2
  exit 2
fi

COLLECTION_DIR="$(cd "${COLLECTION_DIR}" && pwd)"
case "${COLLECTION_DIR}" in
  "${SCRIPT_DIR}/collections/"*) ;;
  *) echo "Collection must be below ${SCRIPT_DIR}/collections" >&2; exit 2 ;;
esac
COLLECTION_NAME="$(basename "${COLLECTION_DIR}")"
if [[ ! "${COLLECTION_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Unsupported collection name: ${COLLECTION_NAME}" >&2
  exit 2
fi
CONTAINER_COLLECTION="/aichallenge/ml_workspace/pilot_net/data_collection/collections/${COLLECTION_NAME}"
cd "${REPO_ROOT}"
CMD="python3 /aichallenge/ml_workspace/pilot_net/data_collection/analyze_collection.py --collection ${CONTAINER_COLLECTION} && cd /aichallenge/ml_workspace/pilot_net && python3 extract_data_from_bag.py --bags-dir ${CONTAINER_COLLECTION}/raw --outdir ${CONTAINER_COLLECTION}/extracted --workers ${EXTRACT_WORKERS:-2} --debug && python3 /aichallenge/ml_workspace/pilot_net/data_collection/prepare_sequence_split.py --collection ${CONTAINER_COLLECTION}" \
  docker compose run --rm --no-deps autoware-command
