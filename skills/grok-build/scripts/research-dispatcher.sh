#!/usr/bin/env bash
# research-dispatcher.sh — Classify research tier, manage cache, dispatch research sub-agents.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="${GROK_BUILD_WORKSPACE:-$(cd "$SKILL_DIR/../.." && pwd)}"
CACHE_DIR="$WORKSPACE_ROOT/memory/research-cache"
TTL_HOURS="${GROK_BUILD_RESEARCH_TTL_HOURS:-48}"

usage() {
  cat <<'EOF'
Usage:
  research-dispatcher.sh detect --task "summary" [--json] [--no-auto-research]
  research-dispatcher.sh dispatch --task "summary" [--tier MEDIUM|HIGH|INTERNAL] [--json] [--no-auto-research]
EOF
}

cmd_detect_tier() {
  local task="$1"
  local lowe
  lower="$(echo "$task" | tr '[:upper:]' '[:lower:]')"

  if echo "$lower" | grep -qE 'self-mod|grok-build skill|yellow.?zone|openclaw workspace'; then
    echo "INTERNAL"
    return
  fi
  if echo "$lower" | grep -qE 'new provider|pricing|external api|integration|security|auth'; then
    echo "HIGH"
    return
  fi
  if echo "$lower" | grep -qE 'refactor|architecture|multi-file|performance|migration'; then
    echo "HIGH"
    return
  fi
  if echo "$lower" | grep -qE 'bugfix|typo|rename|small change|docs only'; then
    echo "INTERNAL"
    return
  fi
  echo "MEDIUM"
}

cache_hash() {
  echo -n "$1" | md5sum 2>/dev/null | awk '{print $1}' || echo -n "$1" | md5 2>/dev/null
}

cache_path_for() {
  local task="$1"
  local tier="$2"
  local hash
  hash="$(cache_hash "${tier}:${task}")"
  echo "$CACHE_DIR/${hash}.md"
}

cache_fresh() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  local age_hours
  age_hours=$(( ($(date +%s) - $(stat -c %Y "$path" 2>/dev/null || stat -f %m "$path")) / 3600 ))
  (( age_hours < TTL_HOURS ))
}

auto_dispatch_research() {
  local task="$1"
  local tier="$2"
  local task_id="$3"
  local cache_path
  cache_path="$(cache_path_for "$task" "$tier")"

  if cache_fresh "$cache_path"; then
    printf '%s\n' "$cache_path"
    return 0
  fi

  mkdir -p "$CACHE_DIR"

  case "$tier" in
    INTERNAL)
      cat >"$cache_path" <<EOF
# Internal Research Cache

Task: $task
Tier: INTERNAL
Generated: $(date -Iseconds)

Use workspace files and MEMORY.md only. No external web research required.
EOF
      printf '%s\n' "$cache_path"
      return 0
      ;;
  esac

  local template=""
  case "$tier" in
    MEDIUM) template="$SKILL_DIR/references/subagent-quick-serper.md" ;;
    HIGH) template="$SKILL_DIR/references/subagent-deep-exa.md" ;;
  esac

  local spawn_dir="$WORKSPACE_ROOT/memory/grok-runs/${task_id}/spawn"
  mkdir -p "$spawn_dir"
  local tier_lowe
  tier_lower="$(echo "$tier" | tr '[:upper:]' '[:lower:]')"
  local spawn_file="$spawn_dir/research-${tier_lower}.json"

  python3 - <<PY
import json, datetime
payload = {
  "type": "research-spawn",
  "tier": "$tier",
  "task": """$task""",
  "template": "$template",
  "cache_path": "$cache_path",
  "task_id": "$task_id",
  "requested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
  "instructions": "Spawn sub-agent using template. Write research output to cache_path as markdown.",
}
with open("$spawn_file", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
print("$spawn_file")
PY
  printf '%s\n' "$cache_path"
}

emit_json() {
  python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]), indent=2, ensure_ascii=False))' "$1"
}

build_payload() {
  local tier="$1"
  local cache_path="$2"
  local task_id="$3"
  local spawned="$4"
  local spawn_request="$5"
  local fresh_flag="false"
  cache_fresh "$cache_path" && fresh_flag="true"

  RD_TIER="$tier" \
  RD_CACHE_PATH="$cache_path" \
  RD_TASK_ID="$task_id" \
  RD_SPAWNED="$spawned" \
  RD_SPAWN_REQUEST="$spawn_request" \
  RD_CACHE_FRESH="$fresh_flag" \
  RD_TTL_HOURS="$TTL_HOURS" \
  python3 - <<'PY'
import json, os
print(json.dumps({
  "tier": os.environ["RD_TIER"],
  "cache_path": os.environ["RD_CACHE_PATH"],
  "cache_fresh": os.environ["RD_CACHE_FRESH"] == "true",
  "spawned": os.environ["RD_SPAWNED"] == "true",
  "spawn_request": os.environ["RD_SPAWN_REQUEST"] or None,
  "ttl_hours": int(os.environ["RD_TTL_HOURS"]),
  "task_id": os.environ["RD_TASK_ID"],
}, ensure_ascii=False))
PY
}

CMD="${1:-}"
shift || true

JSON=false
NO_AUTO=false
TASK=""
TIER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    detect|dispatch) CMD="$1"; shift ;;
    --task) TASK="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --json) JSON=true; shift ;;
    --no-auto-research) NO_AUTO=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$CMD" ]] || { usage >&2; exit 2; }
[[ -n "$TASK" ]] || { echo "--task required" >&2; exit 2; }

mkdir -p "$CACHE_DIR"
TASK_ID="$(cache_hash "$TASK" | cut -c1-12)"
[[ -n "$TIER" ]] || TIER="$(cmd_detect_tier "$TASK")"
CACHE_PATH="$(cache_path_for "$TASK" "$TIER")"
SPAWNED=false
SPAWN_REQUEST=""

if [[ "$CMD" == "dispatch" && "$NO_AUTO" == false ]]; then
  mapfile -t dispatch_lines < <(auto_dispatch_research "$TASK" "$TIER" "$TASK_ID")
  CACHE_PATH="${dispatch_lines[-1]}"
  for line in "${dispatch_lines[@]}"; do
    if [[ "$line" == *"/spawn/research-"*.json ]]; then
      SPAWN_REQUEST="$line"
      SPAWNED=true
    fi
  done
  if cache_fresh "$CACHE_PATH"; then
    SPAWNED=false
  fi
fi

payload="$(build_payload "$TIER" "$CACHE_PATH" "$TASK_ID" "$SPAWNED" "$SPAWN_REQUEST")"

if [[ "$JSON" == true ]]; then
  emit_json "$payload"
else
  echo "$payload"
fi