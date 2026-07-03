# app/archive/

This directory contains **deprecated/archived** components that are no longer
actively maintained. They are kept for reference only.

## Contents

### `nicegui_dashboard/` — Legacy NiceGUI Dashboard

**Status:** Deprecated. Superseded by the Next.js web UI in `webui/`.

The NiceGUI-based dashboard was the primary frontend through Fas 1–4. It has
been replaced by the modern Next.js + React + Tailwind dashboard in
[`../../webui/`](../../webui/), which now uses real backend pipeline data for
all views (Översikt, Analys & Trender, Agentprestanda, Fas 4 Insikter,
Samtalsdetalj, Transkribering, Testlabb).

**Do not add new features here.** New dashboard work goes in `webui/`.

See [docs/WEBUI_MODERNIZATION_PLAN.md](../../docs/WEBUI_MODERNIZATION_PLAN.md)
for the full migration status and plan.

**To run the current dashboard:**
```bash
cd webui && npm install && npm run dev   # → http://localhost:3000
```

**To run the legacy dashboard (not recommended):**
```bash
python -m app.archive.nicegui_dashboard.main
```
