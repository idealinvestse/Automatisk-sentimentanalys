# Security Policy

## Supported Versions

We actively maintain the latest version on the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in **Automatisk sentimentanalys**, please report it responsibly.

**Please do NOT open a public GitHub issue.**

Instead, email the maintainer at: **security@idealinvest.se** (or open a private security advisory on GitHub if available).

We aim to respond within 48 hours and will work with you to understand and resolve the issue.

## Sensitive Data Handling (Call Center Context)

This project is designed for processing sensitive customer service conversations. Key security considerations:

- **Audio & Transcripts**: Never commit real customer audio or transcripts to the repository.
- **API Keys**: `OPENROUTER_API_KEY`, `SENTIMENT_API_KEY`, and Hugging Face tokens are loaded from environment variables or secure secret stores (never hardcoded).
- **PII Redaction**: The pipeline includes early PII redaction for the `callcenter` profile (see `src/pipeline.py` and `pii_redactor.py`).
- **External LLM Calls**: All calls to OpenRouter/Mistral are explicitly logged with the prefix `EXTERNAL LLM CALL`. Transcripts are only sent when explicitly enabled via `--use-mistral-llm` or profile settings.
- **Data at Rest**: Use `.cache/`, `state/`, and `outputs/` (all ignored in `.gitignore`).

## Cloud STT (opt-in)

Cloud speech-to-text sends **raw audio bytes** to a third-party provider (Deepgram). This is **disabled by default**; local Whisper/KB-Whisper runs on your machine unless you explicitly opt in.

### What leaves the machine

When `asr.provider=cloud` (config, API `provider`, or CLI) and `DEEPGRAM_API_KEY` (or `CLOUD_STT_API_KEY`) is set, audio files are uploaded to Deepgram for transcription. Transcript text is returned to your environment. No silent cloud fallback unless `cloud_fallback_local=true`.

### How to enable

1. Install the optional extra: `pip install -e ".[cloud-stt]"`
2. Set `DEEPGRAM_API_KEY` (or `CLOUD_STT_API_KEY`) in the environment or launcher secrets.
3. Set `asr.provider: cloud` in `user_config.yaml`, or pass `"provider": "cloud"` on API `/transcribe` / `/analyze_conversation` requests.

### How to disable

- Set `asr.provider: local` (default), or omit `provider` on API calls.
- Unset or remove `DEEPGRAM_API_KEY` / `CLOUD_STT_API_KEY`.
- Do not enable cloud in launcher settings or web UI.

### Logging

Cloud egress is recorded as a metric (`asr_cloud_egress_total`) and log line `asr_cloud_egress=true provider=deepgram`. **Audio content and transcript bodies are not logged.**

### Call-center audio preprocessing

For the `callcenter` sentiment profile, use `preprocess_mode=callcenter` (or `asr.preprocess_mode: callcenter` in config) for telephone-bandpass and tuned VAD. This is recommended but not forced globally.

- **Production Recommendations**:
  - Always run with `SENTIMENT_API_KEY` set in production.
  - Use a reverse proxy (e.g. nginx, Traefik) with TLS.
  - Consider running inside a private network or with mTLS.
  - Regularly rotate API keys.
  - Enable audit logging for all analysis jobs.

## Dependency Security

We use `ruff`, `mypy`, and GitHub Dependabot (recommended) to monitor for vulnerable dependencies. Run:

```bash
pip install -e ".[dev]"
pre-commit run --all-files   # if configured
```

Thank you for helping keep this project secure!