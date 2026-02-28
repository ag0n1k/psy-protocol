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

if [[ -z "${TARGET}" ]]; then
  usage
  exit 1
fi

REMOTE_DIR="psy-protocol"
echo "Deploying stack on ${TARGET}/${REMOTE_DIR}"

ssh $TARGET "mkdir -p ~/${REMOTE_DIR}/mounts/grafana-data"
ssh $TARGET "mkdir -p ~/${REMOTE_DIR}/mounts/telegram-bot-api-data"
ssh $TARGET "mkdir -p ~/${REMOTE_DIR}/mounts/victoria-metrics-data"

ssh $TARGET "mkdir -p ~/${REMOTE_DIR}/infra/monitoring/grafana/provisioning/datasources/"
ssh $TARGET "mkdir -p ~/${REMOTE_DIR}/infra/monitoring/grafana/provisioning/dashboards/"
ssh $TARGET "mkdir -p ~/${REMOTE_DIR}/infra/monitoring/grafana/dashboards/"

scp -O infra/monitoring/prometheus.yml "${TARGET}:${REMOTE_DIR}/infra/monitoring/prometheus.yml"
scp -O infra/.env.server.example "${TARGET}:${REMOTE_DIR}/.env.example"
scp -O infra/monitoring/grafana/provisioning/datasources/victoriametrics.yml "${TARGET}:${REMOTE_DIR}/infra/monitoring/grafana/provisioning/datasources/victoriametrics.yml"
scp -O infra/monitoring/grafana/provisioning/dashboards/default.yml "${TARGET}:${REMOTE_DIR}/infra/monitoring/grafana/provisioning/dashboards/default.yml"
scp -O infra/monitoring/grafana/dashboards/telegram-bot-api-overview.json "${TARGET}:${REMOTE_DIR}/infra/monitoring/grafana/dashboards/telegram-bot-api-overview.json"
scp -O infra/nginx/default.conf "${TARGET}:${REMOTE_DIR}/infra/nginx/default.conf"


ssh "${TARGET}" "REMOTE_DIR='${REMOTE_DIR}' REMOTE_TMP='${REMOTE_TMP}' bash -s" <<'EOF'
set -euo pipefail

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
docker compose up -d
docker compose ps
EOF

echo "Done."
