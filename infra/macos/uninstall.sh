#!/usr/bin/env bash
set -euo pipefail

PLIST="${HOME}/Library/LaunchAgents/com.psy-protocol.bot.plist"

if [[ ! -f "${PLIST}" ]]; then
    echo "Service is not installed (${PLIST} not found)."
    exit 0
fi

launchctl unload -w "${PLIST}"
rm "${PLIST}"

echo "Service uninstalled."
