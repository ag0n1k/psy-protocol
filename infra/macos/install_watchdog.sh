#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TEMPLATE="${SCRIPT_DIR}/com.psy-protocol.watchdog.plist.template"
PLIST_DEST="${HOME}/Library/LaunchAgents/com.psy-protocol.watchdog.plist"

mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${HOME}/Library/LaunchAgents"

sed -e "s|__PROJECT_DIR__|${PROJECT_DIR}|g" "${TEMPLATE}" > "${PLIST_DEST}"

launchctl unload "${PLIST_DEST}" 2>/dev/null || true
launchctl load -w "${PLIST_DEST}"

echo "Watchdog installed and loaded."
echo "Project dir : ${PROJECT_DIR}"
echo "Plist       : ${PLIST_DEST}"
echo "Interval    : every 120s"
echo ""
echo "Use 'make watchdog-status' and 'make watchdog-logs' to verify."
