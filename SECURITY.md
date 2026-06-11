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