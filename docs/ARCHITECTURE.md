# System Architecture

This document describes the architecture of the Swedish Sentiment Analysis system and the decisions that led to the current design.

## Overview

The system is designed around two main entry points:

1. **CLI** (`src/cli.py`) – local batch processing of texts, audio files, and call conversations.
2. **API** (`src/api/`) – FastAPI REST service for on-demand analysis and batch processing.

Both entry points share the same underlying analysis and transcription engines, but present different interfaces suited to their respective environments.

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  CLI (src/cli.py)          │  API (src/api/)                │
│  • text sentiment            │  • REST endpoints              │
│  • audio transcription       │  • Pydantic validation         │
│  • call analysis             │  • HTTP exception handling     │
│                              │  • batch runners               │
├─────────────────────────────────────────────────────────────┤
│  Core Helpers (src/core/)                                   │
│  • serialization – result formatting, timestamps             │
│  • audio – audio file path resolution                        │
│  • models – dataclasses (Segment, Transcript, etc.)          │
│  • errors – custom exception hierarchy                       │
│  • config – constants and defaults                           │
├─────────────────────────────────────────────────────────────┤
│  Pipeline (src/pipeline.py)                                 │
│  • CallAnalysisPipeline – end-to-end orchestration           │
│  • coordinates transcription → analysis → report             │
├─────────────────────────────────────────────────────────────┤
│  Registry (src/analysis/)                                   │
│  • Analyzer base class + registry                          │
│  • Topological dependency resolution                         │
│  • Error isolation – one failing analyzer doesn't break     │
│    downstream analyzers                                      │
│  • Parametrizable via analyzer_configs                      │
│  • Adapters: sentiment, intent, summary, topics,            │
│    insights, predictive                                      │
├─────────────────────────────────────────────────────────────┤
│  Engines                                                    │
│  • src/sentiment.py – HF transformers sentiment pipeline    │
│  • src/intent.py – intent classification (heuristic/model)  │
│  • src/summarizer.py – abstractive summarization            │
│  • src/topic_modeling.py – topic extraction               │
│  • src/insights.py – action-item extraction               │
│  • src/predictive.py – risk prediction                    │
├─────────────────────────────────────────────────────────────┤
│  Transcription (src/transcription/)                          │
│  • FasterWhisperTranscriber                                │
│  • TransformersTranscriber                                 │
│  • Cached factory – model loaded once per config           │
├─────────────────────────────────────────────────────────────┤
│  Lexicon (src/lexicon.py)                                   │
│  • Swedish sentiment lexicon loader (CSV/TSV)             │
│  • Negation handling (inte, ej, aldrig, …)               │
│  • blend_results_with_lexicon – high-level helper          │
├─────────────────────────────────────────────────────────────┤
│  Utilities                                                  │
│  • src/clean.py – text cleaning per profile               │
│  • src/profiles.py – profile resolution                   │
│  • src/blending.py – model blending                       │
│  • src/diarization.py – speaker diarization               │
│  • src/negation.py – negation detection                   │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Registry + Pipeline Pattern

Rather than hard-coding analysis steps, the system uses a **registry pattern** (`src/analysis/registry.py`) where analyzers declare their dependencies. The pipeline:

1. **Registers** analyzers by name (e.g., `"sentiment"`, `"intent"`, `"summary"`).
2. **Resolves** transitive dependencies automatically (e.g., summary depends on sentiment).
3. **Sorts** them topologically so dependencies execute first.
4. **Isolates errors** – if one analyzer crashes, others still run.

This makes it trivial to add new analyzers without touching the CLI or API code.

### 2. Analyzer Parametrization

`get_registered_analyzers()` accepts an optional `analyzer_configs` mapping:

```python
run_analyzers(
    ctx,
    analyzer_configs={
        "sentiment": {"model_name": "custom-model", "device": "cuda"},
        "intent": {"backend": "model"},
    },
)
```

This decouples analyzer configuration from the registry, letting callers (CLI, API, pipeline) pass settings without modifying global state.

### 3. ASR Model Caching

The `get_transcriber()` factory uses `@lru_cache(maxsize=4)` so that loading a large Whisper model (10–30 seconds) only happens once per unique `(backend, model, device)` configuration. This eliminates the biggest performance bottleneck in batch processing.

### 4. Shared Serialization Helpers

`src/core/serialization.py` centralises result formatting:

- `score_dict()` – safely converts HF pipeline output to a fixed-label mapping.
- `blend_results_with_lexicon()` – high-level lexicon blending that gracefully handles missing files and length mismatches.
- `map_results_to_segment_dicts()` – maps per-text sentiment results back onto transcript segments.

These functions are used by both the CLI and the API, ensuring consistent output formatting and eliminating duplicated logic.

### 5. API Package Structure

(The old monolithic `src/api.py` duplicate has been removed; current code lives in the modular `src/api/` package):

```
src/api/
├── __init__.py          # Re-exports app + create_app
├── app.py               # FastAPI factory + exception handlers
├── schemas.py           # All Pydantic request/response models
├── batch.py             # Generic batch runner (replaces duplicated logic)
├── helpers.py           # Shared ASR helper
└── routers/
    ├── __init__.py
    ├── health.py        # GET /health
    ├── text.py          # POST /analyze
    ├── transcription.py # POST /transcribe, POST /batch_transcribe
    ├── conversation.py  # POST /analyze_conversation, POST /batch_analyze_conversation
    ├── pipeline.py      # POST /analyze_pipeline
    └── scan.py          # POST /scan_process
```

Each router is self-contained, imports shared helpers, and uses `HTTPException` for proper HTTP status codes (400 for validation, 422 for config errors, 500 for processing errors).

### 6. Input Validation

All API endpoints use Pydantic models with `field_validator` for runtime validation:

- `texts` must be non-empty lists.
- `audio_path` and `directory` must exist on the server filesystem.
- Numeric fields have `ge`/`le` bounds (e.g., `batch_size` 1–128).
- `operation` in scan requests is restricted to known values.

Validation runs before any business logic, providing clear error messages to API consumers.

### 7. Race Condition Fix in Scan Processing

The `/scan_process` endpoint previously only wrote its incremental state file after all batches completed. If the process was interrupted mid-run, progress was lost. The new implementation:

1. Persists state **after each batch**.
2. Uses a single shared `run_batch()` helper with timeout support.
3. Gracefully handles `TimeoutError` so that individual file failures don't abort the entire scan.

### 8. Lexicon Blending as a Separate Concern

Lexicon blending lives in `src/lexicon.py` as `blend_results_with_lexicon()` rather than inside the sentiment analyzer. This keeps the analyzer generic (model inference only) and lets the CLI and API decide whether and how to blend, based on user-provided parameters.

## Data Flow

### Text Sentiment Analysis (CLI)

```
texts → clean_texts() → analyze_smart() → blend_results_with_lexicon() → CSV/stdout
```

### Audio Call Analysis (CLI)

```
audio files → resolve_audio_paths() → CallAnalysisPipeline.analyze_audio()
  → get_transcriber() (cached) → transcribe → run_analyzers()
    → sentiment, intent, summary, topics, insights, predictive
  → blend_results_with_lexicon() → CSV/stdout
```

### API Endpoints

| Endpoint | Input | Output |
|---|---|---|
| `POST /analyze` | list of texts | per-text sentiment scores |
| `POST /transcribe` | audio file path | transcript dict |
| `POST /batch_transcribe` | paths/dirs/globs | per-file transcripts |
| `POST /analyze_conversation` | audio file path | transcript + per-segment sentiment |
| `POST /batch_analyze_conversation` | paths/dirs/globs | per-file transcripts + sentiments |
| `POST /analyze_pipeline` | pre-transcribed segments | full analysis (sentiment, intent, summary, …) |
| `POST /scan_process` | directory + pattern | incremental batch processing with state tracking |

## Error Handling Strategy

| Layer | Error Type | Handler |
|---|---|---|
| Pydantic models | Invalid input values | `400 Bad Request` |
| ConfigurationError | Missing models/files | `422 Unprocessable Entity` |
| TranscriptionError | ASR failure | `500 Internal Server Error` |
| AnalysisError | Analyzer crash | `500 Internal Server Error` |
| BaseAnalysisError | Catch-all analysis | `500 Internal Server Error` |
| CLI | Any exception | Error message to console, non-zero exit |

## Extension Points

### Adding a New Analyzer

1. Create a class inheriting from `Analyzer` in `src/analysis/<name>.py`.
2. Implement `analyze(ctx)` returning a dict.
3. Decorate it with `@register_analyzer("<name>")`.
4. Optionally declare dependencies via `requires = [...]`.
5. Add config support by accepting kwargs in `__init__`.

No changes needed in CLI or API – the new analyzer will automatically be picked up by the pipeline.

### Adding a New API Endpoint

1. Create a Pydantic model in `src/api/schemas.py`.
2. Add a router function in `src/api/routers/<domain>.py`.
3. Import and register the router in `src/api/app.py`.

### Adding a New Lexicon

1. Create a CSV or TSV with columns `term` (or `word`) and `polarity` (or `score`/`sentiment`).
2. Pass the path to any CLI command or API endpoint via `--lexicon-file` / `lexicon_file`.

## Backward Compatibility

- The legacy monolithic `src/api.py` has been removed (it was a full duplicate of the refactored `src/api/` package). The current API is `src/api/` (uvicorn src.api:app still works via the package `__init__.py`).
- All existing CLI commands (`sentiment`, `transcribe`, `analyze-call`) behave identically from a user perspective.
- All analyzer constructors still accept the same default arguments; parametrization is opt-in via `analyzer_configs`.
