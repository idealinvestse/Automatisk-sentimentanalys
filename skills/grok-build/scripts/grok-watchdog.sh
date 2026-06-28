#!/usr/bin/env bash
# grok-watchdog.sh — Wrap grok CLI invocations with timeout, heartbeat, and silent-death detection.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="${GROK_BUILD_WORKSPACE:-$(cd "$SKILL_DIR/../.." && pwd)}"

usage() {
  cat <<'EOF'
Usage: grok-watchdog.sh --task-id ID --phase plan|execute [grok args...]

Environment:
  GROK_BUILD_TIMEOUT       Max runtime seconds (default: 3600)
  GROK_BUILD_SILENCE_SECS  Kill if no output for N seconds (default: 600)
  GROK_BUILD_WORKSPACE     Workspace root (default: parent of skills/)

Exit codes:
  0   Success
  124 Timeout (overall or silence)
  125 Silent death (non-zero exit with zero output bytes)
EOF
}

TASK_ID=""
PHASE=""
GROK_BIN="${GROK_BIN:-/root/.grok/bin/grok}"
TIMEOUT_SECS="${GROK_BUILD_TIMEOUT:-3600}"
SILENCE_SECS="${GROK_BUILD_SILENCE_SECS:-600}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task-id) TASK_ID="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --timeout-secs) TIMEOUT_SECS="$2"; shift 2 ;;
    --grok-bin) GROK_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [[ -z "$TASK_ID" || -z "$PHASE" ]]; then
  usage >&2
  exit 2
fi

if [[ $# -lt 1 ]]; then
  echo "grok-watchdog: missing grok command arguments" >&2
  exit 2
fi

RUN_DIR="$WORKSPACE_ROOT/memory/grok-runs/$TASK_ID"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/watchdog.log"
OUTPUT_FILE="$RUN_DIR/grok-output.log"
HEARTBEAT_FILE="$RUN_DIR/heartbeat.json"

log() {
  echo "[$(date -Iseconds)] $*" >>"$LOG_FILE"
}

log "watchdog start phase=$PHASE timeout=${TIMEOUT_SECS}s silence=${SILENCE_SECS}s cmd=$*"

: >"$OUTPUT_FILE"
START_TS="$(date -Iseconds)"
OUTPUT_BYTES=0
LAST_OUTPUT_AT="$START_TS"

write_heartbeat() {
  local pid="${1:-0}"
  cat >"$HEARTBEAT_FILE" <<EOF
{"task_id":"$TASK_ID","phase":"$PHASE","pid":$pid,"started_at":"$START_TS","last_output_at":"$LAST_OUTPUT_AT","output_bytes":$OUTPUT_BYTES,"timeout_secs":$TIMEOUT_SECS,"silence_secs":$SILENCE_SECS}
EOF
}

kill_tree() {
  local pid="$1"
  local sig="$2"
  if kill "-$sig" "$pid" 2>/dev/null; then
    log "sent SIG$sig to pid=$pid"
  fi
  pkill -P "$pid" 2>/dev/null || true
}

set +e
"$GROK_BIN" "$@" >>"$OUTPUT_FILE" 2>&1 &
GROK_PID=$!
set -e

write_heartbeat "$GROK_PID"
ELAPSED=0
SILENCE_ELAPSED=0
EXIT_CODE=0

while kill -0 "$GROK_PID" 2>/dev/null; do
  sleep 5
  ELAPSED=$((ELAPSED + 5))

  # Refresh byte count from file (subshell tee counter is approximate)
  if [[ -f "$OUTPUT_FILE" ]]; then
    local_bytes=$(wc -c <"$OUTPUT_FILE" | tr -d ' ')
    if [[ "$local_bytes" -gt "$OUTPUT_BYTES" ]]; then
      OUTPUT_BYTES="$local_bytes"
      LAST_OUTPUT_AT="$(date -Iseconds)"
      SILENCE_ELAPSED=0
    else
      SILENCE_ELAPSED=$((SILENCE_ELAPSED + 5))
    fi
  else
    SILENCE_ELAPSED=$((SILENCE_ELAPSED + 5))
  fi

  if (( ELAPSED % 30 == 0 )); then
    write_heartbeat "$GROK_PID"
  fi

  if (( ELAPSED >= TIMEOUT_SECS )); then
    log "overall timeout reached (${TIMEOUT_SECS}s)"
    kill_tree "$GROK_PID" TERM
    sleep 30
    kill_tree "$GROK_PID" KILL
    wait "$GROK_PID" 2>/dev/null || true
    write_heartbeat 0
    exit 124
  fi

  if (( SILENCE_ELAPSED >= SILENCE_SECS )); then
    log "silence timeout reached (${SILENCE_SECS}s without output)"
    kill_tree "$GROK_PID" TERM
    sleep 30
    kill_tree "$GROK_PID" KILL
    wait "$GROK_PID" 2>/dev/null || true
    write_heartbeat 0
    exit 124
  fi
done

set +e
wait "$GROK_PID"
EXIT_CODE=$?
set -e

OUTPUT_BYTES=0
if [[ -f "$OUTPUT_FILE" ]]; then
  OUTPUT_BYTES=$(wc -c <"$OUTPUT_FILE" | tr -d '[:space:]')
fi
OUTPUT_BYTES=${OUTPUT_BYTES:-0}

write_heartbeat 0
log "watchdog end exit=$EXIT_CODE output_bytes=$OUTPUT_BYTES"

if [[ "$EXIT_CODE" -ne 0 ]] && [[ ! -s "$OUTPUT_FILE" ]]; then
  log "silent death detected (exit=$EXIT_CODE, zero output)"
  exit 125
fi

exit "$EXIT_CODE"