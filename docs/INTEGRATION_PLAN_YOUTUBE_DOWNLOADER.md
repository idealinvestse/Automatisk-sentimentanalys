# Integrationsplan: YouTube Audio Downloader för Automatisk-sentimentanalys

**Projekt:** https://github.com/idealinvestse/Automatisk-sentimentanalys  
**Syfte:** Full integration av YouTube-ljudnedladdare i både backend (API, CLI, pipeline) och frontend (NiceGUI-dashboard) för att enkelt samla svenska testljud till ASR, sentimentanalys, diarization och call center-intelligens.  
**Bas:** Bygger på det befintliga scriptet `scripts/download_youtube_audio_for_asr.py` (skapad tidigare i projektet).  
**Mål:** Enkel, robust, GDPR-medveten datainsamling direkt från UI/API/CLI med automatisk förberedelse för pipeline.

---

## 1. Översikt och Arkitektur

### Nuvarande läge
- Stark backend med FastAPI (`src/api/`), `CallAnalysisPipeline` (`src/pipeline.py`), ASR i `src/transcription/`, CLI (`src/cli.py`).
- Frontend: NiceGUI-dashboard i `app/nicegui_dashboard/` med WebSocket-stöd för realtid (t.ex. transkription).
- Data: Filbaserat (`data/`, `samples/`, `outputs/`), metadata via JSON + pipeline-resultat.
- Inget befintligt stöd för extern datainsamling från YouTube.

### Önskad integration
- **Core Module:** `src/data_ingestion/youtube_downloader.py` (ny modul eller under `src/transcription/`). 
- **Backend:** Nya endpoints under `/ingest/youtube`, Pydantic-schemas, bakgrundsuppgifter.
- **CLI:** Nytt kommando under `ingest` eller `download-youtube`.
- **Frontend:** Ny flik/sektion "Datainsamling" eller "YouTube Ingest" med formulär, progress och lista över nedladdade samples.
- **Dataflöde:** Automatisk konvertering till 16 kHz mono WAV → metadata → valfri direkt transkription/analys → sparas i `data/ingested/youtube/`.
- **Spårbarhet:** Varje fil får JSON-metadata med källa (YouTube URL, titel, längd, nedladdningstid) + "source": "youtube".

**Teknisk stack som används:**
- yt-dlp (nytt beroende)
- FastAPI + BackgroundTasks (eller framtida Celery/Redis om skalas)
- NiceGUI (ui.input, ui.button, ui.table, ui.progress, ui.audio)
- asyncio.to_thread för sync-operationer
- Befintlig ffmpeg (redan krav)

---

## 2. Detaljerad Implementeringsplan (Fasindelad)

### Fas 0: Förberedelser & Beroenden (1-2 timmar)

1. **Uppdatera beroenden**
   - I `pyproject.toml`:
     ```toml
     [project.optional-dependencies]
     data = [
         "yt-dlp>=2025.6.1",
     ]
     # Eller lägg till i befintlig dev/api
     ```
   - Skapa/uppdatera `requirements-data.txt` eller inkludera i `requirements-dev.txt`.
   - Kör `pip install -e ".[data]"` eller uppdatera Dockerfilen om nödvändigt.

2. **Mappstruktur**
   - Skapa `data/ingested/youtube/` (lägg till i .gitignore om temporära filer).
   - Eventuellt `data/ingested/index.json` för central översikt (valfritt, annars skanna mapp).

3. **Dokumentation**
   - Lägg till varning i UI och docs: "Använd endast offentligt material för test. Respektera YouTubes villkor och upphovsrätt. Endast för internt testdata."

### Fas 1: Core Downloader Modul (3-4 timmar)

Skapa `src/data_ingestion/youtube_downloader.py` (eller `src/transcription/youtube.py`).

**Ny klass (baserad på befintligt script, förbättrad):**

```python
from pathlib import Path
import yt_dlp
import asyncio
from pydantic import BaseModel
from typing import Optional, List
import logging
from src.utils.logging import get_logger  # använd projektets logger

logger = get_logger(__name__)

class DownloadResult(BaseModel):
    success: bool
    file_path: Optional[Path] = None
    metadata: dict = {}
    error: Optional[str] = None

class YouTubeAudioDownloader:
    def __init__(self, output_base: Path = Path("data/ingested/youtube")):
        self.output_base = output_base
        self.output_base.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, playlist: bool = False, convert_to_wav: bool = True, 
                 sample_rate: int = 16000, channels: int = 1) -> DownloadResult | List[DownloadResult]:
        # Anpassad version av befintligt script
        # - Hantera playlist om playlist=True
        # - Använd sanitize_filename
        # - ffmpeg-konvertering med exakta parametrar
        # - Skapa metadata.json per fil
        # - Logga med projektlogger
        # - Returnera DownloadResult eller lista vid playlist
        ...

    async def adownload(self, url: str, **kwargs):
        return await asyncio.to_thread(self.download, url, **kwargs)
```

- Flytta/refaktorera logik från `scripts/download_youtube_audio_for_asr.py` hit.
- Lägg till bra felhantering (yt_dlp.utils.DownloadError, ffmpeg-fel).
- Stöd för progress callbacks om möjligt (yt-dlp har hooks).

**Testbarhet:** Enkel unit test med mockade URLs.

### Fas 2: Backend (API & Schemas) (4-6 timmar)

1. **Pydantic Schemas** (skapa `src/schemas/ingest.py` eller lägg i befintlig):
   ```python
   from pydantic import BaseModel, HttpUrl, Field
   from typing import Optional

   class YouTubeDownloadRequest(BaseModel):
       url: HttpUrl
       playlist: bool = False
       convert_to_wav: bool = True
       sample_rate: int = 16000
       auto_transcribe: bool = False
       auto_analyze: bool = False
       output_subdir: Optional[str] = None

   class DownloadResponse(BaseModel):
       success: bool
       file_path: str
       metadata: dict
       message: str
   ```

2. **Nya Endpoints** i `src/api/` (troligen skapa `src/api/routers/ingest.py` eller lägg till i befintlig router):
   ```python
   from fastapi import APIRouter, BackgroundTasks, Depends
   from src.data_ingestion.youtube_downloader import YouTubeAudioDownloader
   # ...

   router = APIRouter(prefix="/ingest/youtube", tags=["Data Ingestion - YouTube"])

   @router.post("/download", response_model=DownloadResponse)
   async def download_youtube_audio(
       request: YouTubeDownloadRequest,
       background_tasks: BackgroundTasks
   ):
       downloader = YouTubeAudioDownloader()
       result = await downloader.adownload(
           str(request.url), 
           playlist=request.playlist,
           convert_to_wav=request.convert_to_wav
       )
       
       if request.auto_transcribe or request.auto_analyze:
           # Använd befintlig pipeline eller /transcribe endpoint logik
           background_tasks.add_task(
               run_pipeline_on_downloaded, 
               result.file_path, 
               analyze=request.auto_analyze
           )
       
       return DownloadResponse(...)
   ```

   - Liknande endpoint för batch (ta emot lista av URLs).
   - GET `/downloads` – lista alla nedladdade filer + metadata (scanna mapp + läs JSON-filer).
   - DELETE `/downloads/{id}` för cleanup.

3. **Integration med Pipeline**
   - I `src/pipeline.py` eller `CallAnalysisPipeline`: Lägg till `source_metadata` i resultatet.
   - Skapa hjälpfunktion `run_pipeline_on_downloaded(wav_path, auto_analyze=True)` som anropar befintlig transkribe + analyze.

**Tekniska detaljer:**
- Använd `BackgroundTasks` för långa operationer.
- Framöver: Överväg task queue (Celery + Redis) om downloads blir vanliga.
- Validering: Endast youtube.com / youtu.be URLs (pydantic eller custom validator).

### Fas 3: CLI Integration (2-3 timmar)

I `src/cli.py` (som använder Click):

```python
import click
from src.data_ingestion.youtube_downloader import YouTubeAudioDownloader

@click.group(name="ingest")
def ingest_group():
    """Datainsamling och ingestion kommandon."""
    pass

@ingest_group.command("youtube")
@click.argument("url")
@click.option("--playlist", is_flag=True, help="Ladda ner hela spellistan")
@click.option("--no-wav", is_flag=True, help="Behåll originalformat")
@click.option("--auto-analyze", is_flag=True)
def download_youtube(url, playlist, no_wav, auto_analyze):
    """Ladda ner ljud från YouTube och valfritt analysera."""
    downloader = YouTubeAudioDownloader()
    results = downloader.download(url, playlist=playlist, convert_to_wav=not no_wav)
    click.echo(f"Nerladdat: {results}")
    if auto_analyze:
        # Anropa pipeline
        ...
```

- Registrera `ingest_group` i huvud-CLI.
- Lägg till i hjälpen och dokumentation.

### Fas 4: Frontend - NiceGUI Dashboard (5-8 timmar)

**Plats:** `app/nicegui_dashboard/` – lägg till ny komponent eller utöka befintlig main.py / tabs.

**Ny sektion "Datainsamling" (YouTube Ingest):**

```python
# Exempel i en page eller tab
with ui.card().classes('w-full'):
    ui.label('YouTube Ljudnedladdare för Testdata').classes('text-h5')
    
    url = ui.input('YouTube URL eller spellistlänk', placeholder='https://youtube.com/watch?v=...')
    
    with ui.row():
        playlist = ui.checkbox('Spellista')
        convert_wav = ui.checkbox('Konvertera till 16kHz WAV', value=True)
        auto_analyze = ui.checkbox('Transkribera & analysera direkt efter nedladdning')
    
    progress = ui.linear_progress(value=0, show_value=False).classes('w-full')
    
    async def start_download():
        progress.value = 0
        # Anropa API eller direkt modul (bättre via API för konsistens)
        response = await call_api('/ingest/youtube/download', {...})
        # Uppdatera progress via timer eller WebSocket
        ui.notify('Nedladdning klar!')
        refresh_table()
    
    ui.button('Ladda ner', on_click=start_download).classes('bg-primary')

# Tabell över nedladdade filer
table = ui.table(columns=[...], rows=load_downloaded_files())
# Kolumner: Titel, Längd, Källa, Fil, Status, Actions (Transkribera, Analysera, Spela, Ta bort)
```

**Funktioner att implementera:**
- Real-time progress (NiceGUI task runner eller polling + ui.timer).
- Audio preview: `ui.audio(src=...)` eller embedded player.
- Knapp "Transkribera nu" som anropar befintlig `/transcribe` eller WebSocket-transkription.
- "Lägg till i aktivt dataset".
- Lista uppdateras automatiskt efter nedladdning.
- Sök/filter i tabellen.
- Statistik-kort: "Antal YT-samples", "Total speltid", etc.

**Bästa praxis för NiceGUI:**
- Använd `ui.state` eller global store för listan av samples.
- WebSocket för live-uppdateringar om möjligt (redan använt i projektet för transkription).
- Responsive design.

### Fas 5: Datahantering, Pipeline & Verktyg (3 timmar)

- Uppdatera `scripts/prepare_callcenter_data.py` och `generate_testset.py` med stöd för att importera från `data/ingested/youtube/`.
- I `src/pipeline.py`: Säkerställ att `source` fält sparas i JSON-resultat.
- Lägg till exempel-URLs i `samples/youtube_test_urls.txt`.
- Eventuell indexering: Enkel JSON-lista eller utöka befintlig caching.

### Fas 6: Testning, Dokumentation & Polish (3-4 timmar)

- **Tester:** `tests/test_data_ingestion.py` – mock yt-dlp och ffmpeg.
- **Dokumentation:**
  - Uppdatera `README.md` med nya kommandon och UI-beskrivning.
  - Uppdatera `ROADMAP.md` och `UTVECKLINGSPLAN.md`.
  - Lägg till denna plan som `docs/INTEGRATION_PLAN_YOUTUBE_DOWNLOADER.md`.
- **Säkerhet/GDPR:**
  - Logga alla nedladdningar (källa + tid).
  - UI-varningar.
  - Filnamn-sanitizing (redan i script).
- **Prestanda:** Timeout på downloads, hantera stora spellistor (begränsa antal videor).
- **CI:** Uppdatera pre-commit eller GitHub Actions om nödvändigt.

---

## 3. Prioritering & Tidsuppskattning

| Fas | Tid (timmar) | Prioritet | Beroenden |
|-----|--------------|-----------|-----------|
| 0   | 2            | Hög       | -         |
| 1   | 4            | Hög       | 0         |
| 2   | 6            | Hög       | 1         |
| 3   | 3            | Medel     | 1         |
| 4   | 8            | Hög       | 2         |
| 5   | 3            | Medel     | 1-4       |
| 6   | 4            | Låg       | Allt      |

**Totalt:** ~30 timmar (kan delas upp i flera PRs).

**Rekommenderad ordning:** Fas 0 → 1 → 2 (backend först) → 3 (CLI) → 4 (frontend) → 5+6.

---

## 4. Risker & Mitigering

- **Långa nedladdningar:** Använd BackgroundTasks + progress feedback. Begränsa spellistor till 10-20 videor initialt.
- **Upphovsrätt/ToS:** Starka varningar i UI + docs. Fokusera på "testdata" och offentliga källor.
- **Beroende på extern tjänst:** yt-dlp är stabilt men YouTube kan ändra. Ha fallback eller uppdateringsprocess.
- **Resurser:** ffmpeg + yt-dlp kan vara tunga – kör på maskin med bra CPU/IO eller i bakgrund.
- **Skalbarhet:** Börja filbaserat. Senare: Lägg till databas (SQLite) för metadata om volymen ökar.

---

## 5. Framtida Utbyggnader (efter denna integration)

- Stöd för andra källor (Spotify podcasts via yt-dlp, direkta MP3-länkar, RSS).
- Automatisk språkdetektion + filtrering (endast svenska).
- Dashboard: "Rekommenderade svenska källor" med exempel-länkar.
- Batch-upload + bulk-analys.
- Integration med LLM-agent för att föreslå relevanta YouTube-sökningar.
- Export av dataset till Hugging Face eller liknande.

---

## 6. Hur man börjar implementera

1. Skapa branch: `feature/youtube-ingest-integration`
2. Implementera Fas 0 + 1 (core modul + beroenden) → PR
3. Fortsätt med backend (Fas 2)
4. CLI + Frontend parallellt eller sekventiellt
5. Testa end-to-end med riktiga svenska YouTube-länkar (t.ex. SVT, P3-klipp)
6. Merge till main + uppdatera docs

Denna plan är levande – uppdatera den under implementationen.

**Skapad:** 2026-06-19  
**Version:** 1.0  
**Nästa:** Använd denna plan som bas för implementation (se separat Grok Build-prompt).