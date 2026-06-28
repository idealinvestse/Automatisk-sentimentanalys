#!/usr/bin/env bash
# cost-track.sh — Record plan/execute cost telemetry per grok-build task.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="${GROK_BUILD_WORKSPACE:-$(cd "$SKILL_DIR/../.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  cost-track.sh record --task-id ID --phase plan|execute \
    [--input-tokens N] [--output-tokens N] [--cost-usd X] [--model grok-build]

  cost-track.sh finalize --task-id ID
EOF
}

CMD=""
TASK_ID=""
PHASE=""
INPUT_TOKENS=""
OUTPUT_TOKENS=""
COST_USD=""
MODEL="${GROK_BUILD_MODEL:-grok-build}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    record|finalize) CMD="$1"; shift ;;
    --task-id) TASK_ID="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --input-tokens) INPUT_TOKENS="$2"; shift 2 ;;
    --output-tokens) OUTPUT_TOKENS="$2"; shift 2 ;;
    --cost-usd) COST_USD="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$CMD" || -z "$TASK_ID" ]]; then
  usage >&2
  exit 2
fi

MEMORY_DIR="$WORKSPACE_ROOT/memory"
mkdir -p "$MEMORY_DIR"
TELEMETRY="$MEMORY_DIR/cost-telemetry.jsonl"
PENDING="$MEMORY_DIR/grok-pending.json"

python3 - <<'PY' "$CMD" "$TASK_ID" "$PHASE" "$INPUT_TOKENS" "$OUTPUT_TOKENS" "$COST_USD" "$MODEL" "$TELEMETRY" "$PENDING"
import json, datetime, os, sys

cmd, task_id, phase, in_tok, out_tok, cost, model, telemetry, pending = sys.argv[1:]

def load_pending():
    if not os.path.exists(pending):
        return {"tasks": {}}
    with open(pending, encoding="utf-8") as f:
        return json.load(f)

def save_pending(data):
    with open(pending, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

now = datetime.datetime.now(datetime.timezone.utc).isoformat()
data = load_pending()
tasks = data.setdefault("tasks", {})
entry = tasks.setdefault(task_id, {
    "task_id": task_id,
    "model": model,
    "plan_cost_usd": None,
    "execute_cost_usd": None,
    "plan_tokens": {"input": 0, "output": 0},
    "execute_tokens": {"input": 0, "output": 0},
    "started_at": now,
    "completed_at": None,
})

if cmd == "record":
    if phase not in ("plan", "execute"):
        raise SystemExit("phase must be plan or execute")
    in_n = int(in_tok) if in_tok else 0
    out_n = int(out_tok) if out_tok else 0
    cost_val = float(cost) if cost else None
    key = f"{phase}_tokens"
    entry[key] = {"input": in_n, "output": out_n}
    entry[f"{phase}_cost_usd"] = cost_val
    entry["model"] = model
    if not entry.get("started_at"):
        entry["started_at"] = now
    row = {
        "task_id": task_id,
        "phase": phase,
        "model": model,
        "input_tokens": in_n,
        "output_tokens": out_n,
        "cost_usd": cost_val,
        "recorded_at": now,
    }
    with open(telemetry, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_pending(data)
    print(json.dumps(row))

elif cmd == "finalize":
    entry["completed_at"] = now
    save_pending(data)
    print(json.dumps(entry, ensure_ascii=False))
else:
    raise SystemExit(f"unknown cmd: {cmd}")
PY