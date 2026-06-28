---
name: grok-build
description: >
  Plan-and-execute orchestration via native Grok CLI. Research tier detection,
  watchdog-wrapped runs, outbox notifications, and cost tracking. Use for
  multi-step coding tasks requiring plan approval before execute. Triggers on
  grok-build, /grok-build, plan then execute, or Alabama approval flow.
metadata:
  short-description: "Grok plan→execute with watchdog + research"
---

# grok-build

## TLDR

1. `research-dispatcher.sh detect|dispatch --task "..." --json`
2. `compose-plan.sh --task "..." --workdir PATH [--dry-run]`
3. **Plan**: `grok-watchdog.sh --task-id ID --phase plan -- <grok args>`
4. Read `memory/grok-runs/<id>/outbox/plan-ready.json` → send via **message tool** (never shell)
5. Wait for approval: `kör`, `kör på`, `ja`, `go`, `execute`
6. **Execute**: `grok-watchdog.sh --task-id ID --phase execute -- <grok args>`
7. `cost-track.sh record` + `outbox-write.sh notification`
8. Report: `bin/grok-build-cost-report`

**Hard rules**: No `openclaw message send` via shell. No secrets in prompts. Yellow-zone = see `references/self-mod-zones.md`.

---

## Workflow

```text
detect/dispatch → compose-plan → [dry-run?] → grok-watchdog (plan)
  → outbox plan-ready → agent delivers → approval gate
  → grok-watchdog (execute) → cost-track → outbox notification
```

### Phase 0: Research

```bash
skills/grok-build/scripts/research-dispatcher.sh dispatch --task "TASK" --json
```

If `spawned: true`, read `spawn_request` JSON and spawn sub-agent using the referenced template (`subagent-quick-serper.md` or `subagent-deep-exa.md`). Write research to `cache_path`.

Opt-out: `--no-auto-research`

### Phase 1: Compose

```bash
skills/grok-build/scripts/compose-plan.sh \
  --task "TASK" \
  --workdir /root/projects/PROJECT \
  --slug task-slug \
  [--dry-run]
```

- `--dry-run`: writes `memory/grok-plans/<date>-<slug>-PROMPT.md` only; creates `outbox/dry-run-complete.json`
- Otherwise: also writes `outbox/plan-ready.json`

### Phase 2: Plan (Grok)

```bash
GROK_BIN="${GROK_BIN:-/root/.grok/bin/grok}"
skills/grok-build/scripts/grok-watchdog.sh \
  --task-id TASK_ID \
  --phase plan \
  --timeout-secs "${GROK_BUILD_TIMEOUT:-3600}" \
  -m grok-build -p "$(cat memory/grok-plans/...-PROMPT.md)"
```

**Post-run sanity** (exit 125 = silent death):
- Read last 50 lines of `memory/grok-runs/TASK_ID/watchdog.log`
- Verify workdir state before reporting success

Record cost:
```bash
skills/grok-build/scripts/cost-track.sh record --task-id TASK_ID --phase plan \
  --input-tokens N --output-tokens N [--cost-usd X]
```

### Phase 3: Approval gate

Read `memory/grok-runs/TASK_ID/outbox/plan-ready.json` and deliver summary to user via **OpenClaw message tool**. Do not use shell messaging.

### Phase 4: Execute (after approval)

```bash
skills/grok-build/scripts/grok-watchdog.sh \
  --task-id TASK_ID \
  --phase execute \
  -m grok-build -p "Execute the approved plan at ..."
```

```bash
skills/grok-build/scripts/cost-track.sh record --task-id TASK_ID --phase execute ...
skills/grok-build/scripts/cost-track.sh finalize --task-id TASK_ID
skills/grok-build/scripts/outbox-write.sh notification --task-id TASK_ID --text "Execute complete: ..."
```

Agent reads `outbox/notification.json` and sends via message tool.

---

## Workdir selection

| Project | Workdir |
|---------|---------|
| Automatisk-sentimentanalys | `/root/projects/Automatisk-sentimentanalys` |
| OpenClaw workspace | `/root/.openclaw/workspace` |
| MossRouter | `/root/projects/MossRouter` |

---

## Research tiers

| Tier | When | Action |
|------|------|--------|
| INTERNAL | Self-mod, docs, small fixes | Workspace only |
| MEDIUM | Default feature work | Serper sub-agent |
| HIGH | New providers, security, architecture | Exa sub-agent |

See `references/research-augmented-plan.md`.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/grok-watchdog.sh` | Timeout + heartbeat + silent-death detection |
| `scripts/research-dispatcher.sh` | Tier detect, cache, spawn manifest |
| `scripts/compose-plan.sh` | Prompt assembly + dry-run |
| `scripts/outbox-write.sh` | plan-ready / notification / dry-run outbox |
| `scripts/cost-track.sh` | Telemetry + grok-pending.json |
| `bin/grok-build-cost-report` | On-demand cost summary |

---

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `GROK_BUILD_WORKSPACE` | parent of `skills/` | Workspace root |
| `GROK_BUILD_TIMEOUT` | 3600 | Watchdog max seconds |
| `GROK_BUILD_SILENCE_SECS` | 600 | Kill if no output |
| `GROK_BUILD_RESEARCH_TTL_HOURS` | 48 | Research cache TTL |
| `GROK_BIN` | `/root/.grok/bin/grok` | Grok CLI path |

---

## References

- `references/QUICKSTART.md` — copy-paste commands
- `references/prompt-template.md` — plan prompt skeleton
- `references/self-mod-zones.md` — yellow/red zone gates
- `references/subagent-quick-serper.md` — MEDIUM research agent
- `references/subagent-deep-exa.md` — HIGH research agent