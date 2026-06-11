# Contributing to Automatisk sentimentanalys

Thank you for your interest in contributing! This project aims to build a high-quality, production-ready Swedish call center intelligence platform.

## For LLM Coding Agents (Grok, Claude, GPT, Cursor, etc.)

**Please read this first:**

- `docs/LLM_AGENT_GUIDE.md` – Primary guide for agents. Contains architecture, extension patterns, coding standards, and playbooks.
- `docs/ROADMAP.md` – Current status and completed features.
- `README.md` – High-level overview and quickstart.

Agents should follow the patterns described in `LLM_AGENT_GUIDE.md` (especially the Analyzer Registry pattern and graceful degradation principles).

## How to Contribute (Humans)

1. **Fork** the repository and create your feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Install development dependencies**:
   ```bash
   pip install -e ".[dev,diarize]"
   ```
3. Make your changes following the guidelines in `docs/LLM_AGENT_GUIDE.md`.
4. **Run tests and linting**:
   ```bash
   pytest
   ruff check .
   ruff format --check .
   mypy src
   ```
5. **Commit** your changes with clear messages.
6. **Open a Pull Request** against the `main` branch.

## Code Style

- We use **Ruff** for linting and formatting (configured in `pyproject.toml`).
- Type hints are encouraged (`mypy` is part of the dev dependencies).
- Keep functions focused and add docstrings for public APIs.
- New analyzers should be registered in `src/analysis/registry.py`.

## Adding New Features

- **New Analyzer** (e.g. custom intent or emotion): Implement in `src/analysis/` and register it.
- **New ASR Backend**: Add to `src/transcription/` following the existing factory pattern.
- **LLM Improvements**: Update prompts in `src/llm/prompts.py` and schemas in `src/llm/schemas.py`.

## Documentation

- Update `README.md` and relevant files in `docs/` when adding user-facing features.
- For larger changes, update or create entries in `CHANGELOG.md`.

## Reporting Issues

Please use GitHub Issues and include:
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version, GPU?)
- Relevant logs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue with the `question` label or start a discussion.

We appreciate high-quality contributions that improve call center analytics for Swedish users!