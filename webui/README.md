# Call Center Insights – Web UI (Next.js)

Modern ersättning för den NiceGUI-baserade dashboarden i `app/nicegui_dashboard/`.
Bygger mot **samma FastAPI-backend** (`src/api`) – inga backend-ändringar krävs.

Se den övergripande planen: [`docs/WEBUI_MODERNIZATION_PLAN.md`](../docs/WEBUI_MODERNIZATION_PLAN.md).

## Stack

- Next.js 16 (App Router) + React 19 + TypeScript (strict)
- Tailwind CSS v4, design tokens i `src/app/globals.css`
- UI-primitiver i shadcn/ui-stil (Radix + `class-variance-authority`) i `src/components/ui/`
- TanStack Query för server-state, `next-themes` för dark mode, `sonner` för toasts
- `lucide-react` för ikoner, `recharts` för diagram (Fas 1+)

## Köra lokalt

```bash
cd webui
npm install
cp env.example .env.local   # justera NEXT_PUBLIC_API_BASE_URL vid behov
npm run dev                 # http://localhost:3000
```

Backend måste köra separat (samma som för NiceGUI-dashboarden):

```bash
uvicorn src.api:app --port 8000
```

## Scripts

- `npm run dev` – utvecklingsserver (Turbopack)
- `npm run build` / `npm start` – produktionsbygge
- `npm run lint` – ESLint

## Status

Endast `/` (Översikt) är påbörjad som referensimplementation av designsystemet.
Övriga rutter (`/analytics`, `/agents`, `/insights`, `/transcription`, `/testlab`)
är platshållare tills de migreras enligt planen.
