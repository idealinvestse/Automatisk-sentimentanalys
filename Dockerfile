# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf

WORKDIR /app

# System deps (minimal + ffmpeg for ASR and libgomp for ctranslate2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files first for better caching
COPY requirements-min.txt requirements-cli.txt requirements-api.txt requirements.txt ./

# Install Python deps (API + minimal inference + fine-tuning dependencies)
RUN python -m pip install -U pip && \
    pip install -r requirements-min.txt -r requirements-api.txt && \
    pip install -r requirements-cli.txt

# Copy source, configs, data, and docs
COPY src/ ./src/
COPY configs/ ./configs/
COPY data/ ./data/
COPY samples/ ./samples/
COPY docs/ ./docs/
COPY README.md ROADMAP.md ./

# Create cache, outputs, and models directories
RUN mkdir -p /cache/hf /app/outputs /app/models
VOLUME ["/cache/hf"]

EXPOSE 8000

# Default: run API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]