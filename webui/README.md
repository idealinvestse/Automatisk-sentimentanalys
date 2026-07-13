# Call Center Insights – Web UI (Next.js)

Primär dashboard för Automatisk-sentimentanalys.
Bygger mot **samma FastAPI-backend** (`src/api`).

## Stack

- Next.js 16 (App Router) + React 19 + TypeScript (strict)
- Tailwind CSS v4, design tokens i `src/app/globals.css`
- UI-primitiver i shadcn/ui-stil (Radix + `class-variance-authority`) i `src/components/ui/`
- TanStack Query för server-state, `next-themes` för dark mode, `sonner` för toasts
- `lucide-react` för ikoner, `recharts` för diagram

## Köra lokalt

```bash
cd webui
npm install
cp env.example .env.local   # justera NEXT_PUBLIC_API_BASE_URL vid behov
npm run dev                 # http://localhost:3000
```

Backend måste köra separat:

```bash
uvicorn src.api:app --port 8000
```

## Scripts

- `npm run dev` – utvecklingsserver (Turbopack)
- `npm run build` / `npm start` – produktionsbygge
- `npm run lint` – ESLint
- `npm run test:e2e` – Playwright smoke
