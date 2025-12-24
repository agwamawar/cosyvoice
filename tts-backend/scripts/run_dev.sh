#!/bin/bash
# Run development server

set -e

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Default settings for development
export ENVIRONMENT=${ENVIRONMENT:-dev}
export DEBUG=${DEBUG:-true}
export USE_MOCK=${USE_MOCK:-true}
export SERVER_RELOAD=${SERVER_RELOAD:-true}
export LOG_LEVEL=${LOG_LEVEL:-DEBUG}

echo "=== Starting TTS Backend (Development) ==="
echo "Environment: $ENVIRONMENT"
echo "Debug: $DEBUG"
echo "Mock Mode: $USE_MOCK"
echo "Log Level: $LOG_LEVEL"
echo ""

# Run with Poetry
poetry run uvicorn src.main:app \
    --host ${SERVER_HOST:-0.0.0.0} \
    --port ${SERVER_PORT:-8000} \
    --reload \
    --log-level ${LOG_LEVEL,,}
