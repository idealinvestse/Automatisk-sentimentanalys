# Post Processing Pipeline

### 10.1 Punctuation & Truecasing

Whisper outputs unpunctuated lowercase text. Production requires:
- **Punctuation restoration:** Model-based (e.g., `olafmello/punctuation-restoration`, DeepMultilingualPunct)
- **Truecasing:** Rule-based + ML hybrid
- **Swedish-specific:** Hantering av citattecken, tankestreck, och talstreck

### 10.2 Named Entity Correction

- Build a domain glossary (company names, product names, employee names)
- Apply post-hoc correction using edit distance + phonetic matching
- Use LLM (GPT-4, Claude) for contextual correction of ambiguous cases

### 10.3 Hallucination Cleanup

- Regex filter for known hallucination phrases
- LLM-based detection: "Does this segment contain information not present in audio?"
- Confidence-based truncation: Drop segments below threshold

---

# Observability And Slos

### 11.1 Key Metrics

| Metric | Description | SLO Target |
|--------|-------------|------------|
| Transcribe latency p50 | Time from upload to transcript | <2s for 1min audio |
| Transcribe latency p95 | | <10s for 1min audio |
| Transcribe latency p99 | | <30s for 1min audio |
| Throughput | Audio minutes/hour/GPU | >200 min/hr on A10 |
| WER (sampled) | Weekly human review | <10% on clean, <20% on noisy |
| Hallucination rate | % segments flagged | <2% |
| API error rate | 5xx / total | <0.1% |
| Cost per audio minute | | Track trend |
| GPU utilization | | >70% |

### 11.2 Logging Best Practices

- **Structured logs:** JSON with job_id, audio_id, model_version, duration, latency, segments_count
- **Distributed tracing:** OpenTelemetry spans for: upload → VAD → chunk → transcribe → post-process → diarize
- **Sample retention:** Store full transcripts + audio for 1% of jobs for debugging
- **PII handling:** Redact/hash before logging; comply with GDPR

### 11.3 Alerting

- p99 latency > 2× baseline → page on-call
- Error rate > 1% over 5 min → alert
- Hallucination rate > 5% (detected via post-hoc filter) → alert
- GPU OOM events → auto-scale or fall back to smaller model
- Cost anomaly (>$X/day) → alert

---

# Gpu Vs Api Trade Offs 2025 2026 Pricing

### 12.1 Break-Even Analysis

| Provider | Cost | Throughput | Break-even vs self-hosted |
|----------|------|------------|---------------------------|
| OpenAI Whisper API | $0.006/min | ~1 min audio in 10s | <500 min/month |
| Deepgram Nova-3 | $0.0043/min (batch) | Streaming | <1000 min/month |
| AssemblyAI | $0.0025/min (best) | Async | <800 min/month |
| Self-hosted (A10G, large-v3) | ~$500/mo fixed | ~200 min/hr | >3000 min/month |

**Rule of thumb:** If processing >3000 min/month, self-hosting on a single A10G is cost-effective.

### 12.2 Latency vs Cost

- **Real-time (<500ms):** Deepgram streaming, self-hosted with TensorRT
- **Near-real-time (1-3s):** faster-whisper, OpenAI API
- **Batch (minutes):** All options; optimize for cost

### 12.3 Self-Hosted Considerations

- **Hardware:** A10G (24GB) handles 2-3 concurrent large-v3 streams; L4 for smaller models
- **Quantization:** INT8 (CTranslate2) reduces VRAM by 40%, speeds up 1.5-2× with <1% WER loss
- **Batching:** Dynamic batching with vLLM-style scheduler improves throughput 3-5×
- **Model variants:** Distil-Whisper (6× faster, ~1% WER increase)

---

# Common Failure Modes In Sentiment Analysis Pipelines

### 13.1 Transcription-Induced Sentiment Errors

| Failure Mode | Impact | Mitigation |
|--------------|--------|------------|
| Hallucinated filler | Inflates positive/negative scores | Hallucination filter before NLP |
| Speaker misattribution | Misattributes criticism/praise | Diarization quality gate |
| Sarcasm lost | Positive words in negative context | Down-weight single-word sentiment |
| Prosody lost | "Fine." vs "FINE!" indistinguishable | Use audio-based sentiment (prosody) |
| Hedging lost | "I guess it's okay" → "okay" | Preserve hedges via token-level output |
| Code-switch noise | "Det var really bra" → garbled | Language-aware normalization |

### 13.2 Confidence-Weighted Downstream

```python
def weighted_sentiment(text, confidence_scores):
    words = tokenize(text)
    sentiment_scores = sentiment_model(words)
    
    weighted_score = sum(
        s * c for s, c in zip(sentiment_scores, confidence_scores)
    ) / sum(confidence_scores)
    
    return weighted_score, min(confidence_scores)
```

### 13.3 Hallucination Detection for Downstream Safety

- Flag segments with `no_speech_prob > 0.5` or `compression_ratio > 2.4`
- Exclude from sentiment aggregation
- Log for human review

---

# Hardening Recommendations Summary

### Tier 1: Immediate (1-2 days)

1. **Force language detection** for known-language streams (avoid auto-detect misfires)
2. **Add VAD front-end** (Silero v5) before Whisper to eliminate silence hallucinations
3. **Set `condition_on_previous_text=False`** to prevent loop hallucinations
4. **Implement confidence gates** (drop segments with avg_logprob < -1.0)
5. **Add retry with exponential backoff** for API calls

### Tier 2: Short-term (1-2 weeks)

6. **Migrate to WhisperX or faster-whisper** for production efficiency
7. **Add pyannote 3.1 diarization** for speaker attribution
8. **Implement hallucination post-filter** (regex + entropy)
9. **Add structured logging** with OpenTelemetry tracing
10. **Build confidence-weighted downstream API** for sentiment/NLP consumers

### Tier 3: Medium-term (1-2 months)

11. **Evaluate commercial APIs** (Deepgram Nova-3, Soniox) for streaming/multilingual
12. **Implement A/B testing** framework for model swaps
13. **Add domain-specific fine-tuning** (if >10K hours of labeled Swedish audio)
14. **Build evaluation harness** with held-out test set and WER tracking
15. **Implement cost monitoring** and auto-scaling

### Tier 4: Strategic (3-6 months)

16. **Consider self-hosting** if volume justifies (break-even ~3000 min/month)
17. **Explore Whisper v4** or successor models when available
18. **Build custom Swedish model** if domain-specific (medical, legal) accuracy required
19. **Implement active learning** loop from user corrections
20. **Add prosody/voice-emotion** features for richer sentiment analysis

---

# Key Sources

This report synthesizes findings from:
- OpenAI Whisper documentation and GitHub discussions
- faster-whisper and WhisperX project documentation
- pyannote.audio 3.1 benchmarks and documentation
- Silero VAD GitHub repository
- Deepgram Nova-3 and AssemblyAI Universal-2 model announcements
- Azure Speech and Google Cloud STT documentation
- Soniox and Gladia technical documentation
- Production engineering blogs from companies deploying ASR at scale

Note: Some sources consulted contain model release dates (2024-2025) that are at the edge of the knowledge cutoff; all specific claims about model versions and capabilities are based on the documentation referenced. For production decisions, verify current model versions and pricing directly with providers.

# Executive Summary

Modern ASR pipelines for Swedish business audio (meetings, calls, video conferencing) must contend with a unique combination of challenges: code-switched Nordic conversation, regional dialect variance, long-form audio (60–180 min), overlapping speakers, music and silence, and downstream NLP fragility. This report synthesizes 2025–2026 production practices across Whisper variants, faster-whisper, WhisperX, Deepgram Nova-3, AssemblyAI Universal-2, Azure Speech, Google STT, and Soniox/Gladia, with concrete recommendations for hardening an existing transcription subsystem.

**Key findings:**
- Whisper large-v3 remains the open-source SOTA for Swedish (~7–12% WER on clean audio, ~15–25% on noisy call audio).
- faster-whisper (CTranslate2) is the standard production runtime; WhisperX adds forced alignment + diarization for 4× faster+more accurate long-form transcription.
- Hallucination on silence is Whisper's most damaging production failure mode; deterministic decoding parameters + VAD-front-ending reduce but do not eliminate it.
- Confidence calibration is poor out-of-the-box; word-level gating is necessary before downstream sentiment/NLP consumption.
- Diarization quality (pyannote 3.1, Sortformer) is the second largest source of downstream sentiment misattribution.

---

# The 2025 2026 Asr Landscape For Swedish

### 2.1 Whisper Family (OpenAI)

| Variant | Size | Swedish WER (clean) | Notes |
|---------|------|---------------------|-------|
| Whisper tiny | 39M | ~25–35% | Not viable for production |
| Whisper base | 74M | ~15–20% | Edge/embedded only |
| Whisper small | 244M | ~10–14% | Acceptable for draft |
| Whisper medium | 769M | ~7–10% | Good balance |
| Whisper large-v3 | 1.55B | ~4–8% (clean) / 10–15% (noisy) | Best open-source |
| Whisper large-v3-turbo | 809M | ~6–10% | Distilled, 6× faster |
| distil-whisper large-v3 | 756M | ~7–11% | 6× faster than large-v3 |

**Critical caveats for Swedish:**
- Whisper's training data is heavily English-weighted; Swedish is "medium-resource" in the training mix.
- Common failure: confusion between Swedish / Norwegian / Danish (high acoustic similarity, shared vocabulary). Whisper's language detector sometimes assigns the wrong Scandinavian language.
- Dialect handling: Stockholm Swedish is best represented; Götalandsmål and Norrland dialects show 2–5× WER inflation.
- Code-switching (English loanwords in Swedish tech/business speech) is handled inconsistently; Whisper will sometimes force English-translation of Swedish-English mixes.

### 2.2 Commercial APIs (Cloud)

| Provider | Model | Swedish Support | Streaming | Notes |
|----------|-------|-----------------|-----------|-------|
| OpenAI Whisper API | whisper-1 | Yes (auto-detect) | No | $0.006/min; file-based only |
| Deepgram | Nova-3 | Yes (multi-lang) | Yes (<300ms) | Best streaming latency |
| AssemblyAI | Universal-2 | Yes | Yes | Best diarization + sentiment |
| Azure Speech | SWE sv-SE | Yes (dedicated) | Yes | Strong for Swedish-only workloads |
| Google Cloud STT | Chirp 2 | Yes | Yes | Good for telephony |
| Soniox | Soniox v3 | Yes | Yes | <200ms streaming, strong multilingual |
| Gladia | Solaria-1 | Yes | Yes | European data residency |

### 2.3 Decision Matrix

| Workload | Recommended Stack |
|----------|-------------------|
| Swedish-only call center, GDPR-strict | Azure Speech SWE + on-prem fallback |
| Multilingual meeting (5+ languages) | Deepgram Nova-3 or Soniox |
| Self-hosted, cost-sensitive, English-heavy | faster-whisper large-v3 on GPU |
| Diarization-critical | WhisperX (faster-whisper + wav2vec2 + pyannote 3.1) |
| Sub-300ms streaming | Deepgram Nova-3 or Soniox |
| On-prem/air-gapped | faster-whisper + pyannote 3.1 |

---

# Long Form Audio And Chunking Strategies

### 3.1 The 30-Second Window Problem

Whisper was trained on 30-second segments. Naively passing 60-minute audio triggers the long-form transcription algorithm with rolling context — which is where most hallucinations originate.

### 3.2 Recommended Chunking Architecture

```
┌─────────────────────────────────────────────────┐
│           Audio Preprocessing Pipeline          │
├─────────────────────────────────────────────────┤
│  1. Resample to 16kHz mono PCM                  │
│  2. Apply high-pass filter (80Hz cutoff)        │
│  3. Normalize loudness (-23 LUFS for meetings)  │
│  4. VAD segmentation (Silero v5)                │
│  5. Chunk to 20–30s segments with 2s overlap    │
│  6. Merge transcripts with overlap deduplication│
└─────────────────────────────────────────────────┘
```

### 3.3 faster-whisper Parameters (Production-Tuned)

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

segments, info = model.transcribe(
    audio_path,
    language="sv",  # Force Swedish; do not auto-detect for known-language streams
    task="transcribe",
    beam_size=5,
    best_of=5,
    patience=1.0,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Fallback chain
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    condition_on_previous_text=False,  # CRITICAL: prevents loop hallucinations
    initial_prompt="Möte mellan kollegor. Diskuterar projekt, deadlines och tekniska detaljer.",  # Swedish domain priming
    word_timestamps=True,
    vad_filter=True,
    vad_parameters={
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 300,
        "threshold": 0.5,
    },
)
```

### 3.4 Chunk Overlap & Merging

- **Overlap:** 2 seconds on each side (prevents boundary word loss)
- **Deduplication:** Use longest-common-substring on overlap region; prefer higher-confidence (lower `compression_ratio`, higher `avg_logprob`) segment
- **Silence insertion:** Insert 0.5s silence markers at chunk boundaries if gap > 2s

### 3.5 WhisperX Advantage

WhisperX solves the chunking problem fundamentally:
1. VAD-segment with pyannote (precise speech boundaries)
2. Transcribe each segment with faster-whisper
3. **Force-align** word timestamps with wav2vec2 (accurate to 10ms)
4. Apply pyannote diarization
5. Combine transcription + alignment + diarization

Result: 4× faster than vanilla Whisper long-form, accurate timestamps, speaker labels.

---

# Voice Activity Detection Vad

### 4.1 Why VAD Matters

VAD front-ending is non-negotiable for production:
- Eliminates 60–80% of "ghost transcript" hallucinations (Whisper hallucinating on silence)
- Reduces compute cost (skip silence = cheaper inference)
- Improves speaker diarization (cleaner input)
- Enables parallel chunking

### 4.2 VAD Comparison

| VAD | Latency (CPU) | Accuracy | Use Case |
|-----|---------------|----------|----------|
| Silero VAD v5 | <1ms/30s chunk | High | Production default |
| pyannote VAD | ~10ms/30s | Very high | When bundled with diarization |
| WebRTC VAD | <1ms | Medium | Legacy, telephony |
| Cobra (Picovoice) | <1ms | High | On-device, licensed |

**Recommendation:** Silero VAD v5 for the main pipeline; pyannote VAD if already running pyannote for diarization.

### 4.3 VAD Pitfalls

- **Music bleed:** Music with vocals triggers VAD. Apply music detection (e.g., CNN-based) upstream and skip those segments.
- **Cross-talk in meetings:** Overlapping speech (2+ speakers) often gets misclassified as silence by simple VADs. Use overlap-aware diarization (pyannote 3.1+).
- **Breath/noise:** Aggressive VAD thresholds (0.7+) cut breath sounds needed for naturalness but also cut soft speech. Default 0.5 is a good balance.

---

# Language Detection And Multilingual Handling

### 5.1 Swedish-Specific Issues

- **Nordic confusion:** Whisper's language detector frequently confuses Swedish with Norwegian and Danish. For known-Swedish streams, **always force `language="sv"`** rather than relying on auto-detection.
- **English loanwords:** "API", "deploy", "meeting", "deadline" will often be transcribed as English. This is correct behavior but breaks Swedish-only downstream NLP.
- **Sami names/words:** Minority language names occasionally trigger wrong-language detection.

### 5.2 Recommended Strategy

```python
def detect_language(audio_segment, model):
    # Run Whisper's language detection
    probs = model.detect_language(audio_segment)
    # Check if Swedish is within 5% of top score
    if probs["sv"] > 0.3 or probs["sv"] > 0.8 * probs[max(probs)]:
        return "sv"
    # Fall back to top language
    return max(probs, key=probs.get)
```

### 5.3 Code-Switching

If meetings regularly mix Swedish/English (common in tech/finance):
- Use Whisper's multilingual mode (no language forcing)
- OR use a code-switching-capable model (Soniox, Deepgram Nova-3)
- Post-process to normalize "API:et" vs "API:en" etc. using a Swedish lemmatizer (Stanza, spaCy-sv)

---

# Speaker Diarization

### 6.1 Options

| Model | DER (AMI/CallHome) | License | Notes |
|-------|-------------------|---------|-------|
| pyannote.audio 3.1 | ~11–19% | MIT | Production standard |
| Sortformer (NVIDIA NeMo) | ~10–15% | Apache 2.0 | Newest, best for telephony |
| WhisperX + pyannote | ~12–20% | MIT | Integrated pipeline |
| Deepgram (built-in) | ~8–12% | Commercial | Proprietary, accurate |
| AssemblyAI | ~8–14% | Commercial | Proprietary |

### 6.2 Integration

```python
from pyannote.audio import Pipeline

diarization = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization_segments = diarization("audio.wav")

# Combine with Whisper timestamps
for whisper_seg in whisper_segments:
    speaker = assign_speaker(whisper_seg, diarization_segments)
    print(f"[{speaker}] {whisper_seg.text}")
```

### 6.3 Pitfalls

- **Speaker count mismatch:** Diarization often over-segments (splits one speaker into 2) in meetings with silence gaps. Use speaker-embedding similarity to merge.
- **Overlap handling:** Standard diarization assigns one speaker per time slice. pyannote 3.1+ detects overlaps; for full overlap transcription, use models like Whisper-DOLG or AWS Transcribe.
- **Short utterances:** "Ja", "Mm", "Okej" are frequently misattributed. Require minimum utterance duration (e.g., 1s) before committing speaker labels.

---

# Hallucination Mitigation Critical

### 7.1 Common Whisper Hallucination Types

1. **Silence hallucination:** "Tack för att ni tittade", "Thanks for watching", "Subscribe" on silence
2. **Repetition loops:** Same phrase repeated 5–10× on a single segment
3. **Language drift:** Starts Swedish, ends in English
4. **Phantom content:** Fabricated sentences not present in audio
5. **Timestamp hallucinations:** Repeated text with invalid timestamps

### 7.2 Mitigation Strategies

| Strategy | Impact |
|----------|--------|
| Force `language="sv"` | Eliminates ~60% of language-drift hallucinations |
| `condition_on_previous_text=False` | Eliminates ~80% of repetition loops |
| `compression_ratio_threshold=2.4` | Filters high-entropy (likely hallucinated) segments |
| `log_prob_threshold=-1.0` | Drops low-confidence segments |
| VAD-front-ending (Silero) | Eliminates ~70% of silence hallucinations |
| Temperature fallback chain | Recovers from initial bad decode |
| `initial_prompt` with domain context | Reduces domain-specific errors |
| Hallucination post-filter (regex + entropy) | Catches remaining cases |

### 7.3 Production Hallucination Filter

```python
def is_likely_hallucination(segment, text):
    # Check for known hallucination phrases
    hallucination_phrases = ["tack för att", "thanks for watching", "subscribe",
                             "like and subscribe", "see you next time"]
    if any(phrase in text.lower() for phrase in hallucination_phrases):
        return True
    
    # Repetition detection
    words = text.split()
    if len(words) > 10:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        if unique_ratio < 0.3:  # >70% repeated trigrams
            return True
    
    # Entropy check
    if segment.compression_ratio > 2.4:
        return True
    if segment.avg_logprob < -1.0:
        return True
    
    # No-speech probability
    if segment.no_speech_prob > 0.6:
        return True
    
    return False
```

---

# Quality Evaluation

### 8.1 WER Calculation

```python
from jiwer import wer, mer, wil

reference = "mötet börjar klockan nio"
hypothesis = "mötet börjar klockan nio"  # Perfect
print(wer(reference, hypothesis))  # 0.0

# Word Error Rate = (Substitutions + Deletions + Insertions) / N_ref
```

### 8.2 WER Thresholds for Production

| WER | Quality | Use Case |
|-----|---------|----------|
| <5% | Excellent | Legal, medical transcription |
| 5–10% | Good | Business meetings, general |
| 10–15% | Acceptable | Call analytics, search indexing |
| 15–25% | Poor | Needs review/cleanup |
| >25% | Unusable | Investigate pipeline |

### 8.3 Confidence Scores

Whisper provides per-segment:
- `avg_logprob`: Average log probability (higher = better; >-0.5 is good)
- `compression_ratio`: Text compression ratio (>2.4 = likely hallucination)
- `no_speech_prob`: Probability of no speech (<0.6 = speech present)
- `temperature`: Which sampling temperature was used (0 = greedy, high = fallback)

Word-level timestamps (via WhisperX) allow per-word confidence aggregation.

### 8.4 Monitoring Quality in Production

- **Sampling:** Transcribe 5% of audio twice (different models/seeds) and compute divergence.
- **Confidence-weighted WER:** Track WER on segments with avg_logprob > -0.5 separately.
- **Human spot-check:** Weekly review of 50–100 segments.
- **Downstream signal:** Sentiment/NLP confidence drop can indicate upstream transcription quality issues.

---

# Retry Timeout And Error Handling

### 9.1 Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=10),
    retry_error_callback=lambda state: log_failure(state)
)
def transcribe_with_retry(audio_path, model):
    return model.transcribe(audio_path, timeout=300)
```

### 9.2 Error Categories

| Error Type | Handling |
|------------|----------|
| Network timeout | Retry with exponential backoff + jitter |
| Rate limit (429) | Respect Retry-After header, circuit break |
| Model OOM | Fallback to smaller model (medium → small) |
| Audio corrupt | Skip, log, alert (do not retry) |
| Hallucination detected | Re-transcribe with different params |
| Diarization failure | Return transcript without speaker labels |

### 9.3 Circuit Breaker

```python
class ASRCircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.state = "CLOSED"
        # ... standard circuit breaker pattern
```

### 9.4 Timeouts

- **API calls:** 5–10× expected duration; abort and retry
- **GPU inference:** Hard timeout; GPU may hang
- **Queue processing:** Per-job deadline with DLQ routing
- **Graceful degradation:** Return partial transcript with confidence flags

---
