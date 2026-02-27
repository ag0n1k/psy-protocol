#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  infra/deploy_ssh.sh <user@host> [remote_dir]

Description:
  1) Packs infra files into infra/*.tar.gz
  2) Uploads archive via scp
  3) Extracts on remote server
  4) Runs docker compose for:
     - telegram-bot-api
     - victoria-metrics
     - grafana

Notes:
  - Bot app is NOT deployed.
  - If .env does not exist on remote, script copies infra/.env.server.example to .env
    and exits with a reminder to edit credentials first.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGET="${1:-}"
REMOTE_DIR="${2:-~/psy-protocol-infra}"

if [[ -z "${TARGET}" ]]; then
  usage
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_SCRIPT="${ROOT_DIR}/infra/build_bundle.sh"

BUNDLE_PATH="$("${BUILD_SCRIPT}")"
BUNDLE_FILE="$(basename "${BUNDLE_PATH}")"
REMOTE_TMP="/tmp/${BUNDLE_FILE}"

echo "Uploading ${BUNDLE_FILE} to ${TARGET}:${REMOTE_TMP}"
scp "${BUNDLE_PATH}" "${TARGET}:${REMOTE_TMP}"

echo "Deploying stack on ${TARGET}"
ssh "${TARGET}" "REMOTE_DIR='${REMOTE_DIR}' REMOTE_TMP='${REMOTE_TMP}' bash -s" <<'EOF'
set -euo pipefail

mkdir -p "${REMOTE_DIR}"
tar -xzf "${REMOTE_TMP}" -C "${REMOTE_DIR}"
rm -f "${REMOTE_TMP}"

cd "${REMOTE_DIR}"

if [[ ! -f ".env" ]]; then
  cp infra/.env.server.example .env
  echo "Created .env from infra/.env.server.example"
  echo "Please fill TELEGRAM_API_ID and TELEGRAM_API_HASH in ${REMOTE_DIR}/.env, then rerun deploy."
  exit 2
fi

if [[ -z "$(awk -F= '/^TELEGRAM_API_ID=/{print $2}' .env)" || -z "$(awk -F= '/^TELEGRAM_API_HASH=/{print $2}' .env)" ]]; then
  echo "TELEGRAM_API_ID or TELEGRAM_API_HASH is empty in ${REMOTE_DIR}/.env"
  exit 3
fi

docker compose pull
docker compose up -d telegram-bot-api victoria-metrics grafana
docker compose ps
EOF

echo "Done."
