#!/usr/bin/env bash
# Keeps the telegram-bot-api chain alive: OrbStack VM -> containers -> nginx on :8020.
# The bot itself retries forever, but it cannot recover if the Docker stack is down,
# which happens when OrbStack does not come up after a reboot.
# Intentionally no `set -e`: a single failing probe must not abort the whole run.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"

HEALTH_URL="http://127.0.0.1:8020/"
BOT_LABEL="com.psy-protocol.bot"
LOG_FILE="${PROJECT_DIR}/logs/watchdog.log"
LOG_MAX_BYTES=$((1024 * 1024))
WAIT_DOCKER_SECONDS=90
WAIT_API_SECONDS=90

log() {
    printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${LOG_FILE}"
}

rotate_log() {
    local size
    size=$(stat -f%z "${LOG_FILE}" 2>/dev/null || echo 0)
    if [[ "${size}" -gt "${LOG_MAX_BYTES}" ]]; then
        mv -f "${LOG_FILE}" "${LOG_FILE}.1"
    fi
}

# nginx answers 404 on /, so any HTTP status means the chain is reachable.
api_healthy() {
    local code
    code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "${HEALTH_URL}" 2>/dev/null)
    [[ -n "${code}" && "${code}" != "000" ]]
}

docker_healthy() {
    docker info >/dev/null 2>&1
}

wait_for() {
    local probe="$1" deadline="$2"
    local waited=0
    while ! "${probe}"; do
        if [[ "${waited}" -ge "${deadline}" ]]; then
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    return 0
}

mkdir -p "${PROJECT_DIR}/logs"
rotate_log

if api_healthy; then
    exit 0
fi

log "WARN telegram-bot-api unreachable at ${HEALTH_URL}, starting recovery"

if ! docker_healthy; then
    log "INFO docker daemon is down, running 'orb start'"
    # orb start often reports a timeout while the VM actually comes up, so the
    # exit code is ignored and the daemon is probed instead.
    orb start >>"${LOG_FILE}" 2>&1
    if ! wait_for docker_healthy "${WAIT_DOCKER_SECONDS}"; then
        log "ERROR docker daemon still unavailable after ${WAIT_DOCKER_SECONDS}s, giving up this round"
        exit 1
    fi
    log "INFO docker daemon is up"
fi

log "INFO running 'docker compose up -d'"
if ! (cd "${PROJECT_DIR}" && docker compose up -d >>"${LOG_FILE}" 2>&1); then
    log "ERROR 'docker compose up -d' failed"
    exit 1
fi

if ! wait_for api_healthy "${WAIT_API_SECONDS}"; then
    log "ERROR telegram-bot-api still unreachable after ${WAIT_API_SECONDS}s"
    exit 1
fi

log "INFO telegram-bot-api is healthy again, restarting ${BOT_LABEL}"
launchctl kickstart -k "gui/$(id -u)/${BOT_LABEL}" >>"${LOG_FILE}" 2>&1 \
    || log "WARN failed to kickstart ${BOT_LABEL}, the bot has to recover on its own retries"

log "INFO recovery finished"
