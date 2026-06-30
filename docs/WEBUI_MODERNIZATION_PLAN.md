# Webui-modernisering: NiceGUI → Next.js

**Status:** Fas 0 (grund) påbörjad. Se `webui/` för koden.
**Mål:** En snygg, modern, dynamisk dashboard med ett aktuellt JS-ramverk, där all
text, alla knappar, kort och övriga element syns korrekt och är väl formaterade
i både ljust och mörkt läge.

## 1. Bakgrund och beslut

Den befintliga dashboarden (`app/nicegui_dashboard/`, ~25 komponenter) är byggd i
NiceGUI (Python ovanpå Quasar/Vue). Efter avstämning med ägaren av projektet är
beslutet att **byta till ett dedikerat JS-ramverk** snarare än att bara polera
NiceGUI-laget. Detta är ett betydligt större arbete än en ren CSS/UX-uppfräschning,
men möjliggörs av att backend redan är en renodlad REST/WS-API (`src/api`,
FastAPI) som NiceGUI-dashboarden konsumerar via `nicegui_api_client.py` – samma
API kan återanvändas av en ny frontend **utan ändringar i backend**.

Vald stack (Fas 0, redan scaffoldad i `webui/`):

- **Next.js 16 (App Router) + React 19 + TypeScript (strict)**
- **Tailwind CSS v4** med egna design tokens (OKLCH-färger, ljus/mörk)
- **shadcn/ui-mönster** (Radix primitives + `class-variance-authority`) – egna
  kopior i `src/components/ui/` istället för CLI-genererade (ingen TTY i denna
  miljö), men följer exakt samma konventioner så `shadcn` CLI kan användas
  normalt i framtiden (`components.json` kan läggas till vid behov).
- **TanStack Query** för server-state/cache/loading/error
- **next-themes** för dark/light, **sonner** för toasts
- **lucide-react** för ikoner, **recharts** för diagram
- **zustand** + **zod** tillgängliga för klient-state respektive
  schemavalidering av API-svar när det behövs

Den nya appen ligger i `webui/` som ett fristående npm-projekt, helt separat
från Python-backend och från den gamla NiceGUI-dashboarden. Båda kan köras
parallellt tills migrationen är klar.

## 2. Designsystem (gäller alla vyer)

Allt nedan ska vara konsekvent genom hela appen – detta är kärnan i att lösa
"alla texter syns och är formaterade väl, samt knappar, boxar och element":

| Område | Regel |
|---|---|
| Färger | Enbart CSS-variabler i `globals.css` (`--background`, `--foreground`, `--card`, `--primary`, `--muted-foreground`, `--success/warning/destructive`, …). Aldrig hårdkodade hex-färger i komponenter. |
| Kontrast | Alla text/bakgrund-par ska klara WCAG AA (≥4.5:1 för brödtext) i **både** ljust och mörkt läge – verifieras manuellt per komponent vid migrering. |
| Typografi | Fast skala: `text-xl/semibold` (sidtitel) → `text-sm font-medium` (kortrubrik/etiketter) → `text-sm`/`text-xs text-muted-foreground` (brödtext/hint). Ingen fri blandning av storlekar. |
| Kort/boxar | `Card`/`CardHeader`/`CardContent` med fast radius (`--radius`), border och `shadow-sm`. Inga ad-hoc `div`-boxar med egna borders. |
| Knappar | `Button`-varianter: `default` (primär), `secondary`, `outline`, `ghost`, `destructive`. Storlekar `sm/default/lg/icon`. Inga inline-styled knappar. |
| Status/badges | `Badge`-varianter (`success/warning/destructive/secondary/outline`) för API-status, larm, QA-nivåer etc. |
| Loading | `Skeleton` eller spinner i samma layout som den slutgiltiga vyn (ingen layout-hopp). Inga tomma vita ytor under fetch. |
| Tomma/felstater | Återanvändbar `EmptyState`/`ErrorState`-komponent (motsvarar `empty_state.py` i den gamla dashboarden) – tydlig ikon, rubrik, hjälptext, eventuell åtgärdsknapp. |
| Responsivitet | Sidopanel kollapsar under `md`, kort-grid går från 4→2→1 kolumner, tabeller blir horisontellt scrollbara på mobil. |
| Tillgänglighet | Alla ikon-knappar har `aria-label`, fokussynlig ring (`focus-visible:ring`), tangentbordsnavigerbara menyer/flikar (Radix ger detta gratis). |

## 3. Arkitektur

```
webui/
  src/
    app/                 # routes (App Router) – en mapp per flik
    components/ui/       # shadcn-liknande primitiver (Button, Card, Badge, Tabs, ...)
    components/          # dashboard-specifika composites (KpiCard, AppHeader, AppSidebar, …)
    hooks/                # React Query-hooks per backend-endpoint
    lib/api/client.ts     # typed fetch-klient, 1:1 mot nicegui_api_client.py
    lib/nav.ts            # navigationsstruktur (flikar → routes)
```

Routing ersätter NiceGUI:s flikar 1:1:

| Ny route | Legacy-flik | Fas |
|---|---|---|
| `/` | Översikt | 1 |
| `/analytics` | Analys & Trender | 1 |
| `/agents` | Agentprestanda | 1 |
| `/insights` | Fas 4 Insikter | 2 |
| `/calls/[id]` | Samtalsdetalj | 2 |
| `/transcription` | Transkribering | 3 |
| `/testlab` | Testlabb (endast dev) | 3 |

API-klienten (`lib/api/client.ts`) ska få en metod per endpoint som redan
finns i `nicegui_api_client.py` (health, analyze_pipeline, analyze_text,
analyze_conversation, transcribe, batch_transcribe, scan_process,
transcription jobs, agent_performance, semantic_search, hot_topics, qa/score,
alerting/status, status/processes, status/jobs, ws/transcription). Endast
`health`, `analyze_pipeline`, `alerting/status` och `status/*` är inlagda i
Fas 0 – resten läggs till i samband med att respektive vy migreras.

## 4. Faser

### Fas 0 – Grund (gjort i denna session)
- [x] Scaffold Next.js-app (`webui/`), TS strict, ESLint, Tailwind v4.
- [x] Designtokens (ljus/mörk) i `globals.css`.
- [x] UI-primitiver: `Button`, `Card`, `Badge`, `Tabs`, `Skeleton`.
- [x] `AppHeader` (logo, API-status, reload, dark-mode-toggle) + `AppSidebar` (nav).
- [x] Typed API-klient + `useHealth`-hook (TanStack Query).
- [x] Routningsskelett för alla flikar (`/`, `/analytics`, `/agents`, `/insights`,
      `/transcription`, `/testlab`) – migrerade flikar visar riktigt innehåll,
      övriga visar en tydlig "migreras enligt plan"-vy (`ComingSoon`).
- [x] `npm run lint` och `npm run build` gröna.

### Fas 1 – Kärnvyer (högst trafik)
- [ ] `/` Översikt: KPI-kort kopplade till riktig data (kräver ett `/reports`-
      eller motsvarande endpoint, se `demo_provider.py`/`report_ingest.py` för
      datamodell), samtalstabell (`calls_table.py` → `DataTable`/TanStack Table),
      onboarding-banner.
- [ ] `/analytics`: trenddiagram (recharts) motsvarande `analytics_trends.py`,
      `chart_data.py`.
- [ ] `/agents`: agentprestanda-kort/tabell motsvarande `agent_performance.py`.
- [ ] Delad `DataTable`-komponent (sortering, filter, paginering) som ersätter
      `calls_table.py`.

### Fas 2 – Djupdykningsvyer
- [ ] `/insights` (Fas 4 Insikter): hot topics, wordcloud, advanced insights.
- [ ] `/calls/[id]` Samtalsdetalj: QA-scorecard, evidence panel, emotion
      timeline, virtuell transkriptvy (`virtual_transcript.py` → `react-window`
      eller `@tanstack/react-virtual`), LLM judge panel/breakdown.
- [ ] Larmpanel (`alerts_panel.py`, `call_alerts_section.py`) i header + egen vy.

### Fas 3 – Live/interaktiva funktioner
- [ ] `/transcription`: WS-klient mot `/ws/transcription` (ersätter
      `transcription_ws_client.py`), live-loggvy, jobbstatus, ad-hoc-uppladdning.
- [ ] `/testlab` (endast i dev-läge, motsvarande `is_dev_mode()`): pipeline-test
      på JSON-segments (`live_analysis.py`), PII-audit-vy.
- [ ] Toast/notifieringsparitet med `ui_helpers.py` (success/warning/error).

### Fas 4 – Paritet, QA och cutover
- [ ] Sida-vid-sida visuell granskning mot NiceGUI-versionen, flik för flik.
- [ ] Tillgänglighetsgranskning (kontrast, tangentbord, skärmläsare) på samtliga vyer.
- [ ] Responsiv/mobil genomgång.
- [ ] Playwright e2e-rök-test per route; ev. Vitest + React Testing Library för
      kritiska komponenter (DataTable, API-klient).
- [ ] CI: ny job i `.github/workflows/ci.yml` (node setup, `npm ci`, `lint`,
      `build`, ev. tester) för `webui/`.
- [ ] Docker: egen build-stage/`Dockerfile` för `webui` (Next.js standalone
      output) + uppdatering av `docker-compose.nicegui.yml` eller ny compose-fil.
- [ ] Uppdatera `AGENTS.md`/`docs/LLM_AGENT_GUIDE.md`/`README.md` att peka på
      den nya frontenden som primär; flagga NiceGUI-dashboarden som legacy.
- [ ] Avveckla eller arkivera `app/nicegui_dashboard/` när paritet + godkännande
      är klart (görs inte förrän explicit beslut tas).

## 5. Per-komponent-checklista vid migrering

Använd denna checklista för **varje** komponent som flyttas över från
`app/nicegui_dashboard/components/*.py` till `webui/src/`:

1. Identifiera datakällan (vilket API-anrop/vilken state i `state.py`).
2. Lägg till/återanvänd en typad API-metod + React Query-hook.
3. Bygg vyn med befintliga primitiver (`Card`, `Badge`, `Button`, `Tabs`, …);
   skapa ny primitiv i `components/ui/` endast om mönstret saknas.
4. Implementera loading (`Skeleton`), tom (`EmptyState`) och fel (`ErrorState`
   + `sonner`-toast) – inga "blanka" lägen.
5. Kontrollera kontrast i både ljust och mörkt tema.
6. Kontrollera responsivitet (mobil/tablet/desktop).
7. Jämför funktionellt mot den gamla `.py`-komponenten – inget tapp av
   funktionalitet utan explicit beslut.
8. Skriv minst ett rök-test (render utan krasch) eller Playwright-steg.

## 6. Risker / avvägningar

- **Omfattning:** Detta är en fullständig nyskrivning av en ~25-komponents
  dashboard, inte en uppfräschning – betydligt större insats än att bara
  polera NiceGUI-temat. Görs därför fasvis, med båda UI:erna körbara parallellt.
- **Dubbelunderhåll under övergången:** Tills Fas 4 är klar måste vissa
  buggfixar eventuellt göras i båda dashboards. Prioritera att flytta tunga
  features (testlabb, live-transkribering) tidigt om underhållsbördan blir hög.
- **Inga backend-ändringar krävs strukturellt** eftersom `src/api` redan är
  ramverksagnostisk – nya endpoints läggs till vid behov (t.ex. ett samlat
  `/reports`-listendpoint för Översikt om det inte redan finns) men befintliga
  ändras inte.
- **WebSocket-protokoll** (`/ws/transcription`) måste speglas exakt i en TS-
  klient (Fas 3) – verifiera meddelandeformat mot
  `src/api/routers/ws_transcription.py` innan implementation.

## 7. Var koden finns

- Ny frontend: `webui/` (se `webui/README.md` för hur man kör den lokalt).
- Gammal frontend (referens under migreringen): `app/nicegui_dashboard/`.
- Backend (oförändrad): `src/api/`.
