# Task 10 Report: Diarization pyannote path + docs/UI

## Status

**Complete**

## Changes

| File | Summary |
|------|---------|
| `src/transcription/base.py` | `add_diarization(..., diarization_backend=None)` auto-selects pyannote when HF token + importable |
| `tests/test_asr.py` | TDD tests for pyannote preference and heuristic fallback |
| `SECURITY.md` | Cloud STT opt-in section (egress, enable/disable, logging) |
| `docs/API.md` | `DEEPGRAM_API_KEY`, provider params, cloud STT notes |
| `docs/ARCHITECTURE.md` | AsrRouter + diarization backend selection; callcenter preprocess note |
| `docs/WINDOWS_INSTALL.md` | Diarization, cloud STT, callcenter preprocess guidance |
| `launcher/ui_settings_dialog.py` | ASR provider toggle, cloud fallback checkbox, warning copy |
| `webui/src/app/transcription/page.tsx` | Cloud STT opt-in + callcenter preprocess copy |

## Tests

```bash
pytest tests/test_asr.py::TestAddDiarization -v
# 5 passed
```

## Commit

```
feat(asr): prefer pyannote when token present; document cloud STT opt-in
```

## Concerns

- `DiarizationPipeline` reads `HF_TOKEN` only (not `HUGGINGFACE_HUB_TOKEN`) at runtime; selection checks both env vars but pyannote init may need token alias alignment in a follow-up.
- Webui has informational copy only (no provider control); launcher is the config surface for `asr.provider`.
