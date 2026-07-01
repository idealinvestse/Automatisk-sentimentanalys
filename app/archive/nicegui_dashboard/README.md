# Archived: NiceGUI Dashboard

**Status:** Deprecated / archived. Superseded by the Next.js web UI in `webui/`.

This directory contains the legacy NiceGUI-based dashboard that was the
primary frontend through Fas 1–4. It has been replaced by the modern
Next.js + React + Tailwind dashboard in `../../webui/`.

## Why archived?

The NiceGUI dashboard served its purpose but had limitations:
- Python-only ecosystem (no modern JS tooling, component libraries)
- Limited accessibility and responsive design support
- Harder to maintain and extend

The new `webui/` provides:
- Next.js 16 App Router with React 19
- Tailwind CSS v4 + shadcn/ui patterns
- TanStack Query + Table for data management
- Playwright e2e tests + CI integration
- Docker standalone build

## Running the legacy dashboard (not recommended)

```bash
pip install -e ".[dashboard-nicegui]"
python -m app.archive.nicegui_dashboard.main
```

## Migration reference

See `docs/WEBUI_MODERNIZATION_PLAN.md` for the full migration plan and status.
