# Makefile for Automatisk-sentimentanalys
# Provides convenient commands for development, testing, and common tasks.

.PHONY: help install install-dev install-api install-diarize test lint format check clean run-api run-dashboard

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

install-all:  ## Install everything (dev + api + diarize)
	pip install -e ".[dev,api,diarize]"

# =============================================================================
# Quality & Testing
# =============================================================================

test:  ## Run all tests
	pytest -q

test-verbose:  ## Run tests with verbose output
	pytest -v

test-api:  ## Run API tests with coverage
	pytest tests/test_api.py tests/test_api_coverage.py --cov=src/api --cov-fail-under=85

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

# =============================================================================
# Running the application
# =============================================================================

run-api:  ## Start the FastAPI server (development)
	uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

run-dashboard:  ## Start archived NiceGUI dashboard (legacy; use webui/ instead)
	python -m app.archive.nicegui_dashboard.main

run-webui:  ## Start Next.js web UI (primary dashboard)
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