# Transcription v2 – Plan för svenska call center

**Projekt:** Automatisk-sentimentanalys  
**Version:** v0.5-prep → v0.6  
**Senast uppdaterad:** 2026-06-26  
**Mål:** Högre WER/DER på svenska telefon-/VoIP-inspelningar utan att bryta befintliga CLI/API-kontrakt.

## Övergripande principer

- **Bakåtkompatibilitet:** `--preprocess` (bool) och befintliga API-fält behålls.
- **Graceful degradation:** ffmpeg/noisereduce-fel → originalfil; VAD-fel → faster-whisper-default.
- **Call-center-first:** Narrowband, korta interjektioner ("ja"/"nej"), agent/kund-nivåskillnader.
- **Modulärt:** v1 (`preprocess.py`) oförändrad som `basic`; v2 i `preprocess_v2.py`.

## Task-översikt

| ID | Task | Prioritet | Status |
|----|------|-----------|--------|
| **A-1** | Förbättrad VAD + pre-processing pipeline | Hög | ✅ Klar |
| A-2 | ASR benchmark harness + WER/DER mot callcenter-guld | Hög | ⏳ Nästa |
| A-3 | Hotword/prompt-profiler per bransch | Medel | Planerad |
| A-4 | Streaming chunk-optimering för långa samtal (>30 min) | Medel | Planerad |
| B-1 | Dashboard: preprocess_mode i transcription monitor | Medel | Planerad |
| B-2 | API OpenAPI + preset-synk | Låg | Planerad |

---

## A-1 – Förbättrad VAD + pre-processing (KLAR)

### Levererat

1. **`src/transcription/preprocess_v2.py`**
   - Bandpass 100–3400 Hz (telefoni)
   - `dynaudnorm` för jämnare agent/kund-nivå
   - Mjukare noisereduce (`prop_decrease=0.65`)

2. **`src/transcription/vad_callcenter.py`**
   - Silero VAD-parametrar: `threshold=0.35`, kortare `min_speech_duration_ms`, justerad tystnad/padding

3. **`preprocess_mode`-flagga**
   - Värden: `off` | `basic` | `callcenter`
   - Config: `configs/install_defaults.yaml`, `AsrDefaults.preprocess_mode`
   - CLI: `--preprocess-mode callcenter`
   - API: `TranscribeRequest.preprocess_mode`
   - Factory: `resolve_preprocess_mode()` – `--preprocess` + `profile=callcenter` → `callcenter`

4. **Backend-integration**
   - `faster_whisper.py`, `whisperx.py`, `transformers.py` via `prepare_asr_audio()`
   - VAD skickas som `vad_parameters` i faster/whisperx

5. **Tester**
   - `tests/test_preprocess_v2.py`
   - Utökad `tests/test_asr.py` (VAD wiring)

### Användning

```bash
# Rekommenderat för svenska callcenter-inspelningar
python -m src.cli transcribe samples/call.wav \
  --preprocess-mode callcenter \
  --backend faster --language sv

python -m src.cli analyze-call samples/call.wav \
  --preprocess --profile callcenter
```

### Acceptanskriterier (uppfyllda)

- [x] Ny preprocess-kedja utan att bryta `basic`/bool-flaggan
- [x] Callcenter-VAD appliceras endast i `callcenter`-läge
- [x] Temp-filer städas (`PreprocessHandle.cleanup`)
- [x] Enhetstester med mockad ffmpeg
- [x] Dokumentation i denna fil + ARCHITECTURE

---

## A-2 – Benchmark harness (NÄSTA)

Mät WER/DER före/efter A-1 på annoterade callcenter-klipp. Se `src/benchmarks/audio_runner.py`.

**Acceptans:** Rapport i `reports/transcription_v2_benchmark.md` med ≥3 testfiler.

---

## Optimal prompt för nästa fas (A-2)

```
Du arbetar på https://github.com/idealinvestse/Automatisk-sentimentanalys.

Läs docs/TRANSCRIPTION_V2_PLAN.md (A-1 är klar).

Implementera Task A-2: ASR benchmark harness + WER/DER mot callcenter-guld.

1. Utöka src/benchmarks/audio_runner.py med preprocess_mode-jämförelse (off/basic/callcenter)
2. Kör benchmark på tillgängliga samples + dokumentera i reports/transcription_v2_benchmark.md
3. Lägg till pytest för harness-logik (mockad ASR)
4. Uppdatera TRANSCRIPTION_V2_PLAN.md (markera A-2 klar)
5. Commit + push med tydliga meddelanden

Följ LLM_AGENT_GUIDE.md. Graceful degradation. Inga breaking API-ändringar.
```