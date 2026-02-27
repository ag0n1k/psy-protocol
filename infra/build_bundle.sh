#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"
TIMESTAMP="$(date +"%Y%m%d-%H%M%S")"
BUNDLE_NAME="psy-protocol-infra-${TIMESTAMP}.tar.gz"
BUNDLE_PATH="${INFRA_DIR}/${BUNDLE_NAME}"

mkdir -p "${INFRA_DIR}"

tar -czf "${BUNDLE_PATH}" \
  -C "${ROOT_DIR}" \
  docker-compose.yml \
  infra/monitoring \
  infra/.env.server.example \
  infra/README.md

echo "${BUNDLE_PATH}"
