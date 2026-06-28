# grok-build Quickstart

## Daily workflow

```bash
export GROK_BUILD_WORKSPACE=/root/.openclaw/workspace

# 1. Detect / dispatch research
skills/grok-build/scripts/research-dispatcher.sh detect --task "your task" --json

# 2. Compose plan (dry-run to review first)
skills/grok-build/scripts/compose-plan.sh \
  --task "your task" \
  --workdir /root/projects/MyProject \
  --slug my-task \
  --dry-run

# 3. Plan phase (with watchdog)
TASK_ID=<from compose output>
skills/grok-build/scripts/grok-watchdog.sh \
  --task-id "$TASK_ID" --phase plan \
  -m grok-build -p "$(cat memory/grok-plans/...-PROMPT.md)"

# 4. Agent reads outbox/plan-ready.json and sends Telegram summary

# 5. After "kör" — execute phase
skills/grok-build/scripts/grok-watchdog.sh \
  --task-id "$TASK_ID" --phase execute \
  -m grok-build -p "Execute approved plan ..."

# 6. Record cost + notify via outbox
skills/grok-build/scripts/cost-track.sh record --task-id "$TASK_ID" --phase execute --input-tokens 0 --output-tokens 0
skills/grok-build/scripts/outbox-write.sh notification --task-id "$TASK_ID" --text "Execute complete"
```

## Cost report

```bash
bin/grok-build-cost-report
bin/grok-build-cost-report --task abc123
bin/grok-build-cost-report --since 2026-06-01
```

## Flags

| Flag | Effect |
|------|--------|
| `--dry-run` | Write PROMPT.md only, skip Grok + Telegram |
| `--no-auto-research` | Skip spawn manifest creation |
| `GROK_BUILD_TIMEOUT` | Watchdog max seconds (default 3600) |
| `GROK_BUILD_RESEARCH_TTL_HOURS` | Cache TTL (default 48) |