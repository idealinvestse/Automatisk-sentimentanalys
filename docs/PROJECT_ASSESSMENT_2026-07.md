# Projektbedömning — Automatisk Sentimentanalys

**Datum:** 2026-07-01 (uppdaterad samma dag efter triage av testfynd, se §7)
**Metod:** Statisk kodanalys, dokumentgranskning (ROADMAP, LLM_AGENT_GUIDE, CLEANUP_PLAN, WEBUI_MODERNIZATION_PLAN, CHANGELOG, SECURITY), körning av `pytest` (884 tester), `ruff check`, och `npm run lint` i `webui/`.
**Syfte:** Ge ett helikopterperspektiv för att kunna prioritera nästa utvecklingssteg med hög säkerhet.

> **Uppdatering samma dag:** Fynd 3.1 nedan (5 failande golden-tester) har triagerats till grundorsak och åtgärdats. Det var **inte** en logikregression i sentiment/intent-heuristiken — se §7 för full analys och fix. Resten av rapporten (styrkor, övriga svagheter, utvecklingsplan) kvarstår oförändrad.

---

## 1. Sammanfattning (TL;DR)

Detta är ett **ovanligt moget och välarkitekterat open-source-projekt** för sitt stadium — betydligt mer disciplinerat än typiska "vibe-coded" AI-projekt. Styrkorna ligger i arkitektur (registry-mönster, graceful degradation, strikt typning), dokumentationsdisciplin (aktivt underhållen roadmap + städplan) och en ovanligt bred testsvit (884 tester, 879 gröna efter triage — se §7). Svagheterna är koncentrerade till två huvudområden: **(1) en pågående, ofullständig frontend-migrering** (bekräftat: internt kvalitetsprojekt utan extern deadline, se §6) som skapar dubbel UI-yta, och **(2) fortsatt avsaknad av verklig produktionsdata/-drift** trots omfattande "produktionsklar"-infrastruktur (metrics, tracing, Docker) — verklig samtalskorpus är på väg in (se §6).

**Uppdaterat huvudfynd:** Det ursprungliga larmet om 5 failande golden-pipeline-tester i `tests/test_callcenter_golden.py` visade sig **inte** vara en logikregression i sentiment/intent-heuristiken. Grundorsaken var en saknad optional-dependency (`sentencepiece`) i dev-miljön som fick den riktiga HF-sentimentmodellen att krascha vid laddning, kombinerat med ett mockningsgap i testet som inte fångade det fallet. Båda är nu åtgärdade (kod ändrad i denna session). Se §7 för fullständig root-cause-analys — den är kvar i rapporten som ett konkret exempel på hur "testet failar" inte automatiskt betyder "koden är trasig", och som underlag för rekommendation A.1 (reviderad nedan).

---

## 2. Styrkor

### 2.1 Arkitektur & kodkvalitet
- **Registry-mönster för analyzers** (`src/analysis/registry.py`) med autodiscovery — mycket bra extensionspunkt, låg kopplingsgrad. Ny analys = en fil + en decorator, ingen central-fil-ändring krävs.
- **Graceful degradation** är genomgående respekterad: saknad `pyannote.audio` → heuristisk VAD; saknad LLM-nyckel → hoppa över Mistral-steg; enskild analyzer-krasch → logga och fortsätt. Detta är en arkitekturprincip som faktiskt efterlevs i koden, inte bara på papper.
- **Strikt typning & scheman**: Pydantic används konsekvent för LLM I/O (`src/llm/schemas.py`), vilket minskar risken för trasiga LLM-svar som tyst korrumperar pipeline-resultat.
- **Providerabstraktion för LLM**: Mistral/OpenRouter + Groq bakom gemensamt gränssnitt, med dynamisk prissättning via `model_catalog.py` — ovanligt sofistikerat för ett projekt i denna storlek (kostnadsmedvetenhet inbyggd, inte eftertanke).
- **Ren beroendehantering**: Enbart `pyproject.toml` med optional-dependency-grupper (`cli`, `api`, `llm`, `diarize`, `dashboard-nicegui`, `dev`, `training`) — inga föråldrade `requirements*.txt` att tappa bort synk med.
- **Nollträffar på TODO/FIXME/XXX** i `src/` — antingen extremt ren kod eller (mer troligt) disciplinerad vana att lösa saker direkt eller spåra dem i dokument istället för kodkommentarer. Positivt signal oavsett tolkning.

### 2.2 Testtäckning & CI-disciplin
- **884 testfunktioner**, 874 gröna (98.9%) vid denna körning — stor svit för ett soloprojekt/litet team.
- Tydlig märkning av tunga/långsamma tester (`@pytest.mark.slow`, `@pytest.mark.audio`) så CI kan välja snabb vs. full svit.
- Coverage-gate (`fail_under = 65`) med explicit, motiverad omit-lista (ASR/ffmpeg-kod som är svår att enhetstesta) — ärlig om vad som *inte* testas istället för att fejka ett högt tal.
- Separat, hårdare gate för API-säkerhet (`--cov-fail-under=90` på `src/api`) — visar riskbaserad prioritering av testinsats.
- `ruff` (E/F/W/I/N/UP/B/SIM) + `black` + `mypy` konfigurerat med motiverade per-fil-undantag, inte blankettundantag.

### 2.3 Dokumentation & processdisciplin
- **Ovanligt bra "agent-first" dokumentation**: `AGENTS.md` → `docs/LLM_AGENT_GUIDE.md` → `docs/ROADMAP.md` är en tydlig, hierarkisk läsordning för både mänskliga och AI-utvecklare.
- **`docs/CLEANUP_PLAN.md`** är ett faktiskt levande dokument med avbockade faser (dokumentkonsolidering, kodstädning, beroendestädning) — bevis på att teknisk skuld hanteras proaktivt snarare än att ackumuleras tyst.
- Explicit ägarskapstabell för vem som uppdaterar vad och när (PR-checklista för ROADMAP/CHANGELOG-synk).
- CHANGELOG följer Keep a Changelog-konventionen med commit-referenser — spårbarhet är hög.

### 2.4 Säkerhet & integritet (GDPR-kontext)
- PII-redaktion tidigt i pipelinen för `callcenter`-profilen, med Luhn-validering för kortnummer och en växande svensk namn-/adresslista — visar medvetenhet om falska positiver, inte bara närvaro av regex.
- Explicit loggning (`"EXTERNAL LLM CALL"`) vid varje extern LLM-anrop — bra för revisionsspår.
- GDPR-grind för Groq (kräver EU-residency-flagga eller anonymisering före anrop) — regulatorisk medvetenhet inbyggd i kodvägen, inte bara i policy-text.
- `SECURITY.md` med tydlig kontaktväg och konkreta produktionsrekommendationer (TLS, mTLS, key-rotation).

### 2.5 Ny frontend (webui/)
- Modern, väl beprövad stack (Next.js 16, React 19, TS strict, Tailwind v4, TanStack Query) — bra val, inte "shiny new toy"-teknik utan motivering.
- `npm run lint` är i praktiken grönt (1 varning, 0 fel) och `npm run build` rapporteras grönt i planen.
- Genomtänkt designsystem (kontrastkrav WCAG AA, konsekventa tokens, inga hårdkodade hex-färger) dokumenterat *innan* migrering av fler vyer — minskar risk för inkonsekvent UI längre fram.
- QA-process med Playwright MCP (accessibility snapshot + konsolfel + programmatisk kontrastkontroll) är ovanligt rigorös för en dashboard av den här storleken.

---

## 3. Svagheter & risker

### 3.1 ✅ TRIAGERAT: Golden-pipeline-testfel var en miljö-/testgap, inte en logikregression
**Status: åtgärdat samma dag, se §7 för full analys.** Ursprungsfyndet var att 5 av 884 tester i `tests/test_callcenter_golden.py` failade. Efter triage visade det sig att **sentiment-/intent-heuristiken är korrekt** — grundorsaken var att det obligatoriska paketet `sentencepiece` saknades i dev-miljön (listat under `api`/`min`-extras i `pyproject.toml`, men inte under `dev`), vilket fick den riktiga HuggingFace-sentimentmodellen att krascha vid laddning och analyzern att tyst falla tillbaka till ett konstant `"neutral"`-resultat. Testets mockning täckte inte detta fallet. Båda hålen är nu täppta (se §7). Kvarstående lärdom: se rekommendation A.1 (reviderad) i §5.

*(Sekundärt: 3 tester i `test_provision.py` failar fortfarande, men dessa är Linux-specifika sökvägstester som körs i en Windows-devmiljö — miljöartefakter, inte buggar. Bör ha plattforms-skip/marker istället för att tyst faila på Windows.)*

### 3.2 🟡 Pågående frontend-duplicering
Två parallella UI:er existerar samtidigt: den ärvda NiceGUI-dashboarden (`app/archive/nicegui_dashboard/`, ~25 komponenter, trots att den ligger i en mapp som heter `archive/`) och den nya Next.js-appen (`webui/`), som enligt egen plan fortfarande kör på **mockdata** för flera vyer (Översikt, Analys, Agentprestanda, Insikter). Risker:
- Namnet `archive/` signalerar att den gamla dashboarden är avvecklad, men den är fortfarande den enda vägen till *riktig* data för flera vyer tills mockdata bytts ut — detta kan vilseleda nya bidragsgivare/agenter.
- Ej migrerat: virtualiserad transkriptvy, LLM judge-panel, larmpanel i header, wordcloud/avancerade insikter. Dessa är inte kosmetiska — larmpanelen och LLM judge-panelen är operativt viktiga för QA-användningsfallet.
- Två kodbaser att underhålla samtidigt ökar risken för att buggfixar bara görs på ena sidan.

### 3.3 🟡 Dokumentation-kod-drift (mindre men systematiskt)
- `AGENT_CONTEXT.md` beskriver separata routrar för `agent_performance`, `search/semantic`, `qa/score` — i verkligheten ligger alla dessa endpoints i en enda `src/api/routers/pipeline.py` (bekräftat via kodgranskning). Inte fel i sak, men filkartan stämmer inte 1:1 med koden, vilket kan vilseleda en agent som letar efter "rätt fil att ändra i".
- `pipeline.py` är dokumenterat som "< 550 LOC" efter refaktorering (PIPE-01), men filen är faktiskt ~730 rader (21 928 tecken) enligt katalogscanning — konsekvent med CHANGELOG:s egen kommentar "~720 lines" men i strid med ROADMAP-tabellens "< 550 LOC"-påstående. Litet inkonsekvent, men typiskt för snabbt itererande dokumentation.
- `.grok/skills/`-referenser i `AGENT_CONTEXT.md` pekar på Grok Build-specifik tooling som inte är relevant för alla agentmiljöer (t.ex. denna Devin CLI-session) — bra att vara medveten om vilka delar av dokumentationen är verktygsspecifika.

Detta är inte allvarligt i sig, men är en signal om att dokumentationsvolymen (10+ md-filer i root + 27 i `docs/`) börjar bli svår att hålla 100% synkad manuellt, trots den goda processen i `CLEANUP_PLAN.md`.

### 3.4 🟡 Produktionsverklighet vs. produktionsinfrastruktur
Mycket produktionsorienterad infrastruktur finns (Prometheus `/metrics`, OpenTelemetry-stöd, JSON-loggning, `Dockerfile.gpu`, circuit breaker för webhooks) — men enligt egen ROADMAP saknas fortfarande:
- **Verklig, annoterad samtalskorpus** (1000+ produktionssamtal) — all finjustering/utvärdering sker fortfarande på syntetisk/liten data.
- **OpenTelemetry i produktionsdrift** (endast guider/instrumentering, ingen fältvalidering nämns).
- Detta är en klassisk "infrastruktur före produkt-marknadsanpassning"-risk: mycket välbyggd observability för ett system som ännu inte kört i verklig produktionstrafik med riktiga samtal. Inte fel att bygga det, men prioritetsordningen är värd att ifrågasätta (se rekommendationer).

### 3.5 🟢 Mindre punkter
- 21 `ruff`-fel i nuvarande kod (mestadels import-sortering `I001` och `SIM117` nested-with) — låg allvarlighetsgrad, 5 auto-fixbara, men motsäger "ruff är grönt"-antagandet man annars skulle göra av CI-konfigurationen.
- `webui`: React Compiler-varning för `useReactTable` (inkompatibelt bibliotek för memoization) — ofarligt idag, men värt att hålla koll på om TanStack Table uppdateras.
- Ingen `mypy`-körning verifierades i denna granskning (endast konfiguration lästes) — okänt om typfel finns dolt trots `disallow_untyped_defs = false` (relativt tillåtande inställning).

---

## 4. Rekommenderad vidareutvecklingsplan

Kategoriserad efter: **Stabilisera** (skydda befintligt värde) → **Konsolidera** (minska duplicerad yta) → **Väx** (nya kundvärden) → **Härda** (produktionsverklighet). Varje spår har kommentar om varför, risk om ignorerat, och ungefärlig arbetsordning (traversering).

### Spår A — Stabilisera (gör före allt annat)
1. ~~Triagera de 5 failande golden-pipeline-testerna.~~ **✅ Klart, se §7.** Kvarstående uppföljning: säkerställ att CI-miljön alltid installerar `sentencepiece`/`protobuf` (nu tillagt i `dev`-extras i `pyproject.toml`) och överväg samma härdning (mocka på rätt abstraktionsnivå, inte bara yttersta klassmetoden) för andra golden-/integrationstester som rör tunga ML-beroenden (t.ex. emotion-, aspect-analyzers om de har liknande eager-load-mönster).
2. **Lägg till plattforms-marker/skip för `test_provision.py`:s Linux-specifika tester** (`test_venv_python_path_linux` m.fl.) så Windows-utvecklare inte ser falska fails som brus runt riktiga fails. *(Kvarstår — 3 sådana fails observerades även efter fixen ovan.)*
3. **Kör `ruff check --fix`** för de 5 auto-fixbara felen, granska resten manuellt (import-ordning, nested-with). Litet jobb, hög läsbarhetsvinst.
4. **Etablera en lättviktig "definition of done" som inkluderar full testkörning** (inte bara `-x`/delmängd) innan commits till `main` som rör analysheuristik eller beroenden — se §7 för ett konkret exempel på hur ett smalt mock-scope + en saknad optional-dependency kan se ut som en kärnregression om man bara läser testnamn/antal utan att gräva i loggarna.

### Spår B — Konsolidera (minska dubbel yta, störst DX-vinst)
1. **Sätt ett hårt slutdatum eller en tydlig "data cutover"-milstolpe för webui-migreringen.** Byt mockdata mot riktig `/reports`- eller `/analyze_pipeline`-källa för `/`, `/analytics`, `/agents`, `/insights` (redan identifierat i egen plan, §6/Fas 1). Detta är den enda blockeraren för att kunna säga "webui är primär" på riktigt.
2. **Migrera larmpanel och LLM judge-panel** (operativt kritiska för QA-arbetsflödet) innan ytterligare kosmetisk polish på redan migrerade vyer — dessa två saknas fortfarande enligt egen Fas 2-status.
3. **Byt namn eller arkivera på riktigt**: antingen döp om `app/archive/nicegui_dashboard/` till något som signalerar "fortfarande i drift tills webui X är klar" (t.ex. behåll namnet men lägg en README-varningsbanner), eller sätt upp en definitiv avstängningsplan så det inte blir en tredje permanent UI.
4. **Virtualisera transkriptvyn** (`@tanstack/react-virtual`) innan verkliga (långa) samtal används i produktion — nuvarande icke-virtualiserade lista fungerar bara för korta demo-samtal.

### Spår C — Väx (nya kundvärden, i linje med egen ROADMAP v0.5)
1. **Model routing (kostnad/kvalitet)**: `src/llm/routing.py` + `model_catalog` är påbörjat (FAST/BALANCED/DEEP-tiers) — bra ROI om det kopplas till en enkel policy i dashboarden ("spara pengar"-läge vs "max kvalitet"-läge), vilket är ett konkret säljbart värde mot kunder.
2. **Executive Insights-flik + modell-A/B-jämförelse** i webui — redan planerat, naturlig fortsättning efter att insights-vyn har riktig data (se Spår B.1).
3. **Edge AI-MVP-utbyggnad**: `sentimentanalys edge-analyze` finns som CLI; om affärsmålet är on-prem/offline-kunder (t.ex. myndigheter med extra höga datakrav) är detta en differentiator värd att bygga ut mot en tunn UI/rapport-export, snarare än bara CLI.
4. **Multi-språk/marknadsexpansion (DK)** nämns i AGENT_CONTEXT som planerat — bör inte påbörjas förrän svensk kärnheuristik är stabil igen (se Spår A.1), annars dupliceras samma regressionsrisk över fler språk.

### Spår D — Härda (produktionsverklighet, gör i lagom takt — inte allt på en gång)
1. **Skaffa/skapa en verklig (GDPR-säker) annoterad samtalskorpus** innan mer finjusteringsinfrastruktur byggs ut. Just nu riskerar DATA-01-arbetet att optimera mot syntetisk data som inte representerar verkliga samtalsmönster — det är den högst hävstångsinvestering som återstår enligt egen ROADMAP.
2. **Validera observability-stacken (Prometheus/OTEL) mot en pilotkund eller intern lasttest**, inte bara som instrumenteringskod. Bygg inte fler dashboards ovanpå metrics förrän det finns verklig trafik att titta på.
3. **Produktionschecklista-genomgång** (`docs/PRODUCTION_CHECKLIST.md`) bör köras end-to-end mot en faktisk staging-miljö (Docker/VPS) minst en gång innan v0.5 taggas — annars är "produktionsklar" en dokumentationsstatus, inte en verifierad status.
4. **mypy-körning i CI** (om inte redan aktiv) för att fånga typfel som `ruff` inte täcker — konfigurationen finns men kördes inte i denna granskning; värt att bekräfta att den faktiskt är en gate och inte bara lokal möjlighet.

---

## 5. Traversering — rekommenderad ordning

```
A.1 (triagera golden-test-regression)
  → A.2, A.3, A.4 (städa CI-brus, parallellt, lågt jobb)
    → B.1 (riktig data i webui) ──┬─→ B.2 (larm/LLM-judge-paneler)
                                   └─→ C.2 (Executive Insights ovanpå riktig data)
      → B.3 (arkivbeslut NiceGUI) → B.4 (virtualiserad transkriptvy)
        → D.1 (verklig korpus) → C.1 (model routing i produkt) → C.3 (Edge AI-utbyggnad)
          → D.2 (observability-validering) → D.3 (produktionschecklista end-to-end)
            → C.4 (marknadsexpansion DK)
```

**Kärnprincip för prioritering:** Skydda det som redan fungerar (A) innan du minskar duplicerad yta (B), innan du bygger nytt ovanpå en stabil grund (C), innan du hävdar produktionsmognad utåt (D). Att bygga vidare på C eller D-spår medan A är obekräftat riskerar att förstärka en dold regression över fler ytor (fler språk, fler kunder, fler dashboards som visar fel siffror med hög säkerhet).

---

## 6. Öppna frågor — besvarade av ägaren (2026-07-01)

- **Är de 5 golden-test-failen kända sedan tidigare?** Nej, ny upptäckt. → Triagerad och åtgärdad, se §7. Detta bekräftar att det inte finns en dold, känd-men-ohanterad regression sedan tidigare — bra signal, men understryker vikten av att köra full testsvit regelbundet (Spår A.4).
- **Finns en verklig kund/pilotanvändare som väntar på webui?** Nej — internt kvalitetsprojekt utan extern deadline. Detta sänker tidspressen på Spår B (konsolidering) något, men ändrar inte prioritetsordningen: dubbel UI-yta är fortfarande en underhållsrisk oavsett deadline, och bör lösas innan Spår C (nya funktioner) läggs ovanpå webui.
- **Finns tillgång till en verklig, GDPR-godkänd samtalskorpus?** Kommer längre fram eller redan ikväll. → Spår D.1 kan därmed flyttas upp i prioritet så snart korpusen finns; se reviderad kommentar nedan.

**Konsekvens för traverseringen i §5:** Eftersom D.1 (verklig korpus) nu är nära förestående snarare än obestämt blockerad, bör den behandlas som en **parallell snarare än sekventiell** aktivitet — påbörja valideringsarbete (`scripts/validate_domain_corpus.py`, dataimport-pipeline) så fort korpusen landar, utan att vänta på att Spår B (webui-konsolidering) är helt klart. De två spåren är oberoende av varandra (data-kvalitet vs. UI-duplicering) och kan drivas parallellt av olika personer/sessioner om resurser finns.

---

## 7. Appendix — Root-cause-analys: golden-pipeline-testfel (triagerat 2026-07-01)

**Symptom:** 5 av 884 tester failade i `tests/test_callcenter_golden.py`, samtliga med `assert False` på raden som kontrollerar `has_positive_sentiment` (eller motsvarande negativ/intent-kontroll).

**Undersökning:**
1. Isolerad körning (`pytest tests/test_callcenter_golden.py::... -v --tb=long`) visade i loggarna:
   `ERROR src.analysis.sentiment: Sentiment analysis failed in adapter: Failed to initialize sentiment model 'cardiffnlp/twitter-xlm-roberta-base-sentiment': Converting from SentencePiece and Tiktoken failed...`
2. Kodspårning: `SentimentAnalyzer.analyze()` (`src/analysis/sentiment.py`) anropar `self._get_pipeline()`, som via `get_pool().get_sentiment_pipeline(...)` konstruerar en riktig `SentimentPipeline` (`src/sentiment.py`). Den konstruktorn laddar **eagerly** en riktig HuggingFace-modell (`self._nlp = pipeline("sentiment-analysis", ...)`) — inte lat vid `.analyze()`-anrop.
3. Testets `mock_heavy_backends`-fixture monkeypatchade endast `SentimentPipeline.analyze`, inte konstruktorn/`_get_pipeline()`. När modellkonstruktionen kraschade fångades felet av `SentimentAnalyzer.analyze()`s egna `try/except`, som **tyst faller tillbaka till ett konstant `{"label": "neutral", "score": 0.0}`** för varje segment — mocken nåddes aldrig.
4. Miljökontroll: `python -c "import sentencepiece"` → `ModuleNotFoundError`. XLM-RoBERTa-tokenizern (som cardiffnlp-modellen bygger på) kräver `sentencepiece` för konvertering till en snabb tokenizer. Paketet är listat i `pyproject.toml` under `[project.optional-dependencies].api`/`.min`, men **saknas i `.dev`**, vilket är den grupp `docs/LLM_AGENT_GUIDE.md` rekommenderar agenter att installera (`pip install -e ".[dev,diarize]"`).
5. Verifiering: `pip install sentencepiece` → samtliga 7 tester i filen blev gröna. Avinstallerade `sentencepiece` igen och verifierade att den härdade mocken (steg nedan) gör testerna gröna **oavsett** om paketet finns eller ej (utom det avsiktligt omockade `@pytest.mark.slow`-testet, som korrekt kräver riktiga beroenden).

**Slutsats:** Ingen regression i sentiment-/intent-heuristiken. Detta var en kombination av (a) en miljö/beroende-lucka i `dev`-extras och (b) ett för grunt mock-scope i testet som gjorde felet svårt att diagnostisera (assertion-fel om affärslogik, istället för ett tydligt "modul saknas"-fel).

**Åtgärder genomförda i denna session:**
- `pyproject.toml`: La till `sentencepiece>=0.1.99` och `protobuf>=3.20.0` i `[project.optional-dependencies].dev` med en kommentar som förklarar varför.
- `tests/test_callcenter_golden.py`: Härdade `mock_heavy_backends`-fixturen att även patcha `SentimentAnalyzer._get_pipeline` (returnerar en `_FakeSentimentPipeline`), så testet är isolerat från riktig modell-laddning oavsett miljö.
- Verifierat: full testsvit går från **8 failande → 3 failande** (874→879 gröna av 884). De 3 kvarvarande är `test_provision.py`:s Linux-specifika sökvägstester som körs i en Windows-devmiljö (separat, lågprioriterat städjobb, se Spår A.2).

**Rekommendation framåt:** Gör en snabb genomgång av andra tester som mockar analyzer-klasser med liknande "adapter kring eager-loaded ML-modell"-mönster (t.ex. emotion-, aspect-, role-analyzers om de har egna HF-pipelines) för att se om samma mock-scope-brist finns där. Överväg också att lägga till ett tydligt, tidigt `pytest`-varningsmeddelande (eller en `conftest.py`-kontroll) om kritiska optional-dependencies saknas, istället för att låta det yttra sig som en förvirrande affärslogik-assertion långt nedströms.
