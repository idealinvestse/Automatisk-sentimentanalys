#!/usr/bin/env bash
# compose-plan.sh — Detect research, compose plan prompt, optional dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="${GROK_BUILD_WORKSPACE:-$(cd "$SKILL_DIR/../.." && pwd)}"

usage() {
  cat <<'EOF'
Usage:
  compose-plan.sh --task "summary" --workdir PATH [--slug name] [--dry-run] [--no-auto-research]

Writes:
  memory/grok-plans/<date>-<slug>-PROMPT.md
  memory/grok-runs/<task-id>/outbox/ (plan-ready or dry-run-complete)
EOF
}

TASK=""
WORKDIR=""
SLUG=""
DRY_RUN=false
NO_AUTO=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --slug) SLUG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=true; shift ;;
    --no-auto-research) NO_AUTO=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$TASK" && -n "$WORKDIR" ]] || { usage >&2; exit 2; }

DATE="$(date +%Y-%m-%d)"
if [[ -z "$SLUG" ]]; then
  SLUG="$(echo "$TASK" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//;s/-$//' | cut -c1-40)"
fi

DISPATCH_ARGS=(dispatch --task "$TASK" --json)
[[ "$NO_AUTO" == true ]] && DISPATCH_ARGS+=(--no-auto-research)

DISPATCH_JSON="$("$SCRIPT_DIR/research-dispatcher.sh" "${DISPATCH_ARGS[@]}")"
TIER="$(echo "$DISPATCH_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["tier"])')"
CACHE_PATH="$(echo "$DISPATCH_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["cache_path"])')"
TASK_ID="$(echo "$DISPATCH_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["task_id"])')"
SPAWN_REQUEST="$(echo "$DISPATCH_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("spawn_request") or "")')"

PLANS_DIR="$WORKSPACE_ROOT/memory/grok-plans"
mkdir -p "$PLANS_DIR"
PROMPT_FILE="$PLANS_DIR/${DATE}-${SLUG}-PROMPT.md"
TEMPLATE="$SKILL_DIR/references/prompt-template.md"

{
  echo "# Grok Build Plan Prompt"
  echo ""
  echo "**Date**: $DATE"
  echo "**Workdir**: \`$WORKDIR\`"
  echo "**Task**: $TASK"
  echo "**Research tier**: $TIER"
  echo "**Research cache**: \`$CACHE_PATH\`"
  echo ""
  if [[ -n "$SPAWN_REQUEST" ]]; then
    echo "> Research sub-agent spawn pending: \`$SPAWN_REQUEST\`"
    echo "> Agent must complete research before plan phase if cache is stale."
    echo ""
  fi
  echo "---"
  echo ""
  if [[ -f "$TEMPLATE" ]]; then
    sed "s|{{WORKDIR}}|$WORKDIR|g; s|{{TASK}}|$TASK|g; s|{{TIER}}|$TIER|g; s|{{DATE}}|$DATE|g" "$TEMPLATE"
  else
    echo "## Goal"
    echo ""
    echo "$TASK"
    echo ""
    echo "## Context"
    echo "- Workdir: $WORKDIR"
    echo "- Research tier: $TIER"
  fi
  echo ""
  echo "---"
  echo ""
  echo "## Research Notes"
  echo ""
  if [[ -f "$CACHE_PATH" ]]; then
    cat "$CACHE_PATH"
  else
    echo "_Research cache not yet populated._"
  fi
} >"$PROMPT_FILE"

if [[ "$DRY_RUN" == true ]]; then
  "$SCRIPT_DIR/outbox-write.sh" dry-run-complete --task-id "$TASK_ID" --prompt-path "$PROMPT_FILE"
  echo "DRY_RUN prompt=$PROMPT_FILE task_id=$TASK_ID"
  exit 0
fi

SUMMARY="Plan prompt ready: $SLUG
Workdir: $WORKDIR
Tier: $TIER
File: $PROMPT_FILE

Reply kör / kör på / ja to execute."
"$SCRIPT_DIR/outbox-write.sh" plan-ready --task-id "$TASK_ID" --text "$SUMMARY"
echo "PROMPT=$PROMPT_FILE TASK_ID=$TASK_ID TIER=$TIER"