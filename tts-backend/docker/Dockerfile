# =============================================================================
# TTS Backend Production Dockerfile
# =============================================================================
# Multi-stage build for optimized production image
# Uses NVIDIA CUDA runtime for GPU inference on RunPod
# =============================================================================
#
# RUNPOD DEPLOYMENT NOTES:
# -------------------------
# For RunPod pods, use this image directly with the following setup:
#
# 1. Network Storage:
#    - Mount model weights at /models (or set MODEL_PATH env var)
#    - Mount voices directory at /app/voices for persistence
#    - Models should be pre-downloaded to network storage
#
# 2. Required Environment Variables:
#    - MODEL_PATH=/runpod-volume/models/cosyvoice  (path to model weights)
#    - COSYVOICE_DEVICE=cuda                       (use GPU)
#    - COSYVOICE_DTYPE=float16                     (for faster inference)
#    - API_KEYS=your-api-key-here                  (for authentication)
#    - API_AUTH_ENABLED=true                       (enable auth in prod)
#
# 3. Optional Environment Variables:
#    - DOWNLOAD_MODEL=true                         (download model on startup)
#    - MODEL_URL=https://...                       (URL to download model from)
#    - SERVER_WORKERS=1                            (keep at 1 for GPU)
#    - LOG_LEVEL=INFO                              (logging verbosity)
#    - VOICES_DIRECTORY=/runpod-volume/voices      (persistent voice storage)
#
# 4. RunPod Template Settings:
#    - Container Image: your-registry/tts-backend:latest
#    - Container Disk: 20GB minimum
#    - Volume Disk: 50GB+ for models
#    - Volume Mount Path: /runpod-volume
#    - Expose HTTP Port: 8000
#
# 5. Example RunPod Start Command:
#    (Default entrypoint handles startup automatically)
#
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip && \
    pip install poetry==1.7.1

# Configure Poetry
RUN poetry config virtualenvs.create false

WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install production dependencies only
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Get site-packages location for copying
RUN python -c "import site; print(site.getsitepackages()[0])" > /tmp/site-packages-path

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS runtime

LABEL maintainer="TTS Backend Team"
LABEL description="Production image for TTS Backend with CosyVoice on CUDA"

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # App settings
    ENVIRONMENT=prod \
    DEBUG=false \
    USE_MOCK=false \
    LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    # Server settings
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000 \
    SERVER_WORKERS=1 \
    # Model settings
    MODEL_PATH=/app/models/cosyvoice \
    MODEL_DEVICE=cuda \
    COSYVOICE_USE_MOCK=false \
    # Paths
    VOICES_DIRECTORY=/app/voices

# -----------------------------------------------------------------------------
# Install Python 3.11 and system dependencies
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python
    python3.11 \
    python3.11-dev \
    python3-pip \
    # Audio processing
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# -----------------------------------------------------------------------------
# Copy Python packages from builder
# -----------------------------------------------------------------------------
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# -----------------------------------------------------------------------------
# Set working directory and create structure
# -----------------------------------------------------------------------------
WORKDIR /app

RUN mkdir -p /app/src /app/voices /app/models /app/logs

# -----------------------------------------------------------------------------
# Copy application code
# -----------------------------------------------------------------------------
COPY src/ ./src/
COPY docker/entrypoint.sh /entrypoint.sh

# Make entrypoint executable
RUN chmod +x /entrypoint.sh

# -----------------------------------------------------------------------------
# Create non-root user for security
# -----------------------------------------------------------------------------
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# -----------------------------------------------------------------------------
# Expose port
# -----------------------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# -----------------------------------------------------------------------------
# Entrypoint and default command
# -----------------------------------------------------------------------------
ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
