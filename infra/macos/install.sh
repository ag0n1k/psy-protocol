#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_PYTHON="${HOME}/venv/psy/bin/python3"
if [[ ! -f "${VENV_PYTHON}" ]]; then
    echo "Error: virtualenv not found at ${VENV_PYTHON}" >&2
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi

TEMPLATE="${SCRIPT_DIR}/com.psy-protocol.bot.plist.template"
PLIST_DEST="${HOME}/Library/LaunchAgents/com.psy-protocol.bot.plist"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${HOME}/Library/LaunchAgents"

sed \
    -e "s|__PROJECT_DIR__|${PROJECT_DIR}|g" \
    -e "s|__VENV_PYTHON__|${VENV_PYTHON}|g" \
    "${TEMPLATE}" > "${PLIST_DEST}"

launchctl load -w "${PLIST_DEST}"

echo "Service installed and loaded."
echo "Project dir : ${PROJECT_DIR}"
echo "Python      : ${VENV_PYTHON}"
echo "Plist       : ${PLIST_DEST}"
echo ""
echo "Use 'make service-status' to verify."
