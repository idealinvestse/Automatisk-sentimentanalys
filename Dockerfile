# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf

WORKDIR /app

# System dependencies
# - ffmpeg: required for audio preprocessing (--preprocess)
# - libgomp1: required by some ML libraries (ctranslate2, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata first for better layer caching (Full pyproject.toml migration – 2026-06-28 audit)
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY launcher/ ./launcher/
COPY app/ ./app/

# Install Python dependencies via optional-deps
# Note: For GPU support, use a CUDA base image or install torch with CUDA separately.
# Example GPU image: FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN python -m pip install -U pip && \
    pip install -e ".[min,asr,api,cli,install]"

# Copy remaining assets
COPY configs/ ./configs/
COPY data/ ./data/
COPY samples/ ./samples/
COPY docs/ ./docs/
COPY ROADMAP.md ./

# Create necessary directories
RUN mkdir -p /cache/hf /app/outputs /app/models /app/state

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app /cache/hf
USER appuser

VOLUME ["/cache/hf", "/app/outputs", "/app/state"]

EXPOSE 8000

# Default command: Start API server
# For GPU workloads, ensure the host has NVIDIA drivers and use --gpus all when running the container.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Usage examples:
#   CPU: docker build -t sentimentanalys .
#   GPU: Use a CUDA base image + docker run --gpus all ...
# =============================================================================
