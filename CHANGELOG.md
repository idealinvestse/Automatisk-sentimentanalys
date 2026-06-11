# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-06-12

### Added
- `diarize` optional dependency group in `pyproject.toml` (`pyannote.audio>=3.1.0` + `torchaudio>=2.2.0`) to properly support Speaker Diarization feature.
- `CHANGELOG.md` for tracking releases and changes.
- Improved cross-platform Quickstart and Installation instructions in README (unified `pip install -e "[cli,api,diarize]"` recommendation + Docker example + Hardware Requirements).
- Hardware Requirements section in README (GPU/VRAM guidance for ASR and diarization).

### Changed
- Bumped version from 0.3.0 → **0.4.0** to better reflect the advanced Call Center Intelligence features (Fas 1–3, Mistral LLM integration, full `CallAnalysisPipeline`, API v0.4.0 capabilities).
- Updated project description and keywords in `pyproject.toml` to include "call-center" and "diarization".
- Clarified that `pyannote.audio` now has an official installation path (previously missing from dependency declarations).

### Fixed
- Missing dependency declaration for the Speaker Diarization feature (critical bug fix).
- Inconsistent installation experience between Windows-focused scripts and modern Python packaging (`pyproject.toml` optional deps).

### Documentation
- Added prominent Quickstart section with tiered installation (basic CLI → full call center + diarization → API).
- Added note about `ffmpeg` requirement for `--preprocess`.
- Better guidance on when to use `--use-mistral-llm` and privacy implications.

## [0.3.0] - Previous

- Initial public structure with core sentiment, ASR (faster-whisper), CLI, basic pipeline and evaluation framework.
- Introduction of `CallAnalysisPipeline`, analysis registry, profiles and Streamlit dashboard.
- Mistral / OpenRouter LLM integration (Fas 3) with structured output and caching.

See git history for earlier development details.