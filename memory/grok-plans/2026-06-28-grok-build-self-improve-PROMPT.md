# Grok Build Plan Prompt: grok-build Skill Self-Improvement

**Date**: 2026-06-28  
**Workdir**: `/root/.openclaw/workspace`  
**Branch**: `feat/grok-build-self-improve`  
**Classification**: INTERNAL (yellow-zone self-mod)

## Goal

Implement R1–R9 from the approved self-improvement plan: watchdog, auto-research spawn manifests, outbox notifications, dry-run, cost tracking, tests, and docs for `skills/grok-build/`.

## Context

- Target: `skills/grok-build/` on OpenClaw workspace
- Known bugs: silent Grok failures, manual research despite SKILL.md promise, shell-based messaging
- PR: separate from skill-bundle

## Requirements

See approved plan R1–R9. Implementation delivered in branch `feat/grok-build-self-improve`.

## Verification

```bash
bash skills/grok-build/tests/run-all.sh
! grep -r "openclaw message send" skills/grok-build/
bin/grok-build-cost-report --days 7
```

## Plan Summary

Watchdog + outbox + auto-research + dry-run + cost report. Reply **kör** to execute on OpenClaw.