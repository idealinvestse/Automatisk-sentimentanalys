#!/usr/bin/env bash
# outbox-write.sh — Write notification/plan-ready payloads for OpenClaw agent delivery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="${GROK_BUILD_WORKSPACE:-$(cd "$SKILL_DIR/../.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  outbox-write.sh plan-ready  --task-id ID --text "summary" [--recipient ID]
  outbox-write.sh notification --task-id ID --text "message" [--recipient ID] [--priority normal|high]
  outbox-write.sh dry-run-complete --task-id ID --prompt-path PATH
EOF
}

TYPE=""
TASK_ID=""
TEXT=""
RECIPIENT="${GROK_BUILD_RECIPIENT:-438805461}"
PRIORITY="normal"
PROMPT_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    plan-ready|notification|dry-run-complete) TYPE="$1"; shift ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --text) TEXT="$2"; shift 2 ;;
    --recipient) RECIPIENT="$2"; shift 2 ;;
    --priority) PRIORITY="$2"; shift 2 ;;
    --prompt-path) PROMPT_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$TYPE" || -z "$TASK_ID" ]]; then
  usage >&2
  exit 2
fi

OUTBOX_DIR="$WORKSPACE_ROOT/memory/grok-runs/$TASK_ID/outbox"
mkdir -p "$OUTBOX_DIR"

case "$TYPE" in
  plan-ready)
    [[ -n "$TEXT" ]] || { echo "--text required" >&2; exit 2; }
    FILE="$OUTBOX_DIR/plan-ready.json"
    python3 - <<PY
import json, datetime
payload = {
  "type": "plan-ready",
  "channel": "telegram",
  "recipient": "$RECIPIENT",
  "text": """$TEXT""",
  "task_id": "$TASK_ID",
  "approval_keywords": ["kör", "kör på", "ja", "go", "execute"],
  "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open("$FILE", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print("$FILE")
PY
    ;;
  notification)
    [[ -n "$TEXT" ]] || { echo "--text required" >&2; exit 2; }
    FILE="$OUTBOX_DIR/notification.json"
    python3 - <<PY
import json, datetime
payload = {
  "type": "notification",
  "channel": "telegram",
  "recipient": "$RECIPIENT",
  "text": """$TEXT""",
  "priority": "$PRIORITY",
  "task_id": "$TASK_ID",
  "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open("$FILE", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print("$FILE")
PY
    ;;
  dry-run-complete)
    [[ -n "$PROMPT_PATH" ]] || { echo "--prompt-path required" >&2; exit 2; }
    FILE="$OUTBOX_DIR/dry-run-complete.json"
    python3 - <<PY
import json, datetime
payload = {
  "type": "dry-run-complete",
  "task_id": "$TASK_ID",
  "prompt_path": "$PROMPT_PATH",
  "written_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
with open("$FILE", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print("$FILE")
PY
    ;;
esac