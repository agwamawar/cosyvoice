#!/bin/bash
# =============================================================================
# TTS Backend Entrypoint Script
# =============================================================================
# Handles:
# - Environment variable validation
# - Directory creation
# - Model download (if configured)
# - Server startup
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HOST=${SERVER_HOST:-0.0.0.0}
PORT=${SERVER_PORT:-8000}
WORKERS=${SERVER_WORKERS:-1}
RELOAD=${SERVER_RELOAD:-false}
MODEL_PATH=${MODEL_PATH:-/app/models/cosyvoice}
VOICES_DIR=${VOICES_DIRECTORY:-/app/voices}
LOGS_DIR=${LOGS_DIRECTORY:-/app/logs}
DOWNLOAD_MODEL=${DOWNLOAD_MODEL:-false}
MODEL_URL=${MODEL_URL:-""}

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo "[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# -----------------------------------------------------------------------------
# Check required environment variables
# -----------------------------------------------------------------------------
check_required_vars() {
    local missing=false

    # In production, certain vars are required
    if [ "$ENVIRONMENT" = "prod" ]; then
        if [ -z "$API_KEYS" ] && [ "$API_AUTH_ENABLED" = "true" ]; then
            log_warn "API_KEYS not set but auth is enabled"
        fi
    fi

    # Check if using mock mode
    if [ "$USE_MOCK" = "true" ] || [ "$COSYVOICE_USE_MOCK" = "true" ]; then
        log_info "Running in mock mode - GPU/model not required"
    else
        # Check if CUDA is available (for non-mock mode)
        if command -v nvidia-smi &> /dev/null; then
            log_info "NVIDIA GPU detected:"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
        else
            log_warn "No NVIDIA GPU detected - consider using USE_MOCK=true"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Create necessary directories
# -----------------------------------------------------------------------------
create_directories() {
    log_info "Creating directories..."

    # Create directories if they don't exist
    mkdir -p "$MODEL_PATH" 2>/dev/null || true
    mkdir -p "$VOICES_DIR" 2>/dev/null || true
    mkdir -p "$LOGS_DIR" 2>/dev/null || true

    log_info "Directories ready"
}

# -----------------------------------------------------------------------------
# Download model if requested
# -----------------------------------------------------------------------------
download_model() {
    if [ "$DOWNLOAD_MODEL" = "true" ]; then
        # Check if model already exists
        if [ -d "$MODEL_PATH" ] && [ "$(ls -A $MODEL_PATH 2>/dev/null)" ]; then
            log_info "Model already exists at $MODEL_PATH"
            return 0
        fi

        if [ -z "$MODEL_URL" ]; then
            log_warn "DOWNLOAD_MODEL=true but MODEL_URL not set"
            return 0
        fi

        log_info "Downloading model from $MODEL_URL..."
        
        # Download model (supports HuggingFace, direct URLs, etc.)
        if command -v wget &> /dev/null; then
            wget -q --show-progress -O /tmp/model.tar.gz "$MODEL_URL" || {
                log_error "Failed to download model"
                return 1
            }
        elif command -v curl &> /dev/null; then
            curl -L -o /tmp/model.tar.gz "$MODEL_URL" || {
                log_error "Failed to download model"
                return 1
            }
        else
            log_error "Neither wget nor curl available for download"
            return 1
        fi

        # Extract model
        log_info "Extracting model..."
        tar -xzf /tmp/model.tar.gz -C "$MODEL_PATH" || {
            log_error "Failed to extract model"
            return 1
        }

        rm -f /tmp/model.tar.gz
        log_info "Model downloaded and extracted to $MODEL_PATH"
    fi
}

# -----------------------------------------------------------------------------
# Print startup banner
# -----------------------------------------------------------------------------
print_banner() {
    echo "============================================="
    echo "  TTS Backend - CosyVoice 3"
    echo "============================================="
    echo "  Environment: ${ENVIRONMENT:-dev}"
    echo "  Host:        $HOST"
    echo "  Port:        $PORT"
    echo "  Workers:     $WORKERS"
    echo "  Mock Mode:   ${USE_MOCK:-false}"
    echo "  Model Path:  $MODEL_PATH"
    echo "============================================="
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    print_banner
    check_required_vars
    create_directories
    download_model

    log_info "Starting TTS Backend server..."

    # Check if we should use reload mode (development)
    if [ "$RELOAD" = "true" ]; then
        log_info "Running in development mode with hot reload..."
        exec uvicorn src.main:app \
            --host "$HOST" \
            --port "$PORT" \
            --reload
    else
        log_info "Running in production mode..."
        exec uvicorn src.main:app \
            --host "$HOST" \
            --port "$PORT" \
            --workers "$WORKERS"
    fi
}

# Run main function or execute passed command
if [ $# -eq 0 ]; then
    main
else
    # If arguments are passed, execute them (allows overriding CMD)
    exec "$@"
fi
