# Makefile for Automatisk-sentimentanalys
# Provides convenient commands for development, testing, and common tasks.

.PHONY: help install install-dev install-api install-diarize install-training install-semantic test lint format check clean run-api run-dashboard intent-validate intent-benchmark

help:  ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install:  ## Install with CLI profile (basic usage)
	pip install -e ".[cli]"

install-dev:  ## Install with dev + diarize (recommended for development)
	pip install -e ".[dev,diarize]"

install-api:  ## Install with API profile (includes core ML + REST deps)
	pip install -e ".[api]"

install-diarize:  ## Install diarization support only
	pip install -e ".[diarize]"

install-training:  ## Install intent fine-tuning and evaluation dependencies
	pip install -e ".[training,min,dev]"

install-semantic:  ## Install semantic search dependencies
	pip install -e ".[semantic]"

install-all:  ## Install everything (dev + api + diarize + semantic)
	pip install -e ".[dev,api,diarize,semantic]"

# =============================================================================
# Quality & Testing
# =============================================================================

test:  ## Run all tests
	pytest -q

test-verbose:  ## Run tests with verbose output
	pytest -v

test-api:  ## Run API tests with coverage (≥90% on src/api)
	python -m pytest \
		tests/test_api_smoke.py \
		tests/test_api.py \
		tests/test_api_coverage.py \
		tests/test_api_services.py \
		tests/test_api_security.py \
		tests/test_api_upload.py \
		tests/test_scan_logic.py \
		tests/test_alerting_router.py \
		tests/test_transcription_jobs.py \
		tests/test_ws_ticket_minimal.py \
		tests/test_path_validation.py \
		tests/test_router_errors.py \
		tests/test_batch.py \
		tests/test_status_api.py \
		tests/contracts/test_api_error_contract.py \
		-q --cov=src/api --cov-report=term-missing --cov-fail-under=90

lint:  ## Run ruff linting
	ruff check .

format:  ## Format code with ruff
	ruff format .

check:  ## Run lint + format check + mypy
	ruff check .
	ruff format --check .
	mypy src --ignore-missing-imports

pre-commit:  ## Run pre-commit on all files
	pre-commit run --all-files

intent-validate:  ## Validate train/validation intent corpora for leakage and balance
	python scripts/validate_intent_corpus.py data/intent_train.jsonl --min-rows 200 --min-per-intent 20 --disjoint-from data/intent_val.jsonl
	python scripts/validate_intent_corpus.py data/intent_val.jsonl --min-rows 50 --min-per-intent 5

intent-benchmark:  ## Benchmark heuristic intent backend on the fixed validation set
	python scripts/benchmark_intent.py --val-file data/intent_val.jsonl --backend heuristic --output reports/intent_baseline.json --min-macro-f1 0.75

# =============================================================================
# Running the application
# =============================================================================

run-api:  ## Start the FastAPI server (development)
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

run-dashboard:  ## Start Next.js web UI (primary dashboard)
	cd webui && npm run dev

run-webui:  ## Start Next.js web UI (alias for run-dashboard)
	cd webui && npm run dev

run-cli-help:  ## Show CLI help
	python -m src.cli --help

# =============================================================================
# Maintenance
# =============================================================================

clean:  ## Remove Python cache and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov dist build *.egg-info

update-deps:  ## Update dependencies (use with caution)
	pip install --upgrade pip
	pip install -e ".[dev,diarize]" --upgrade

# =============================================================================
# LLM / Evaluation
# =============================================================================

eval-llm:  ## Run LLM quality evaluation
	python -m src.evaluate llm-quality

# Default target
.DEFAULT_GOAL := help