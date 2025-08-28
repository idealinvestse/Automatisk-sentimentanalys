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
COPY requirements-min.txt requirements-cli.txt requirements-api.txt ./

# Install Python deps (API + minimal inference; CLI optional)
RUN python -m pip install -U pip && \
    pip install -r requirements-min.txt -r requirements-api.txt && \
    pip install -r requirements-cli.txt

# Copy source
COPY src/ ./src/
COPY samples/ ./samples/
COPY README.md ./

# Create cache and outputs directories
RUN mkdir -p /cache/hf /app/outputs
VOLUME ["/cache/hf"]

EXPOSE 8000

# Default: run API server
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
