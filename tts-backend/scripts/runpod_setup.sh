#!/bin/bash
# RunPod Setup Script
# Run this after cloning the repo on a new RunPod instance

set -e

echo "=== RunPod TTS Backend Setup ==="

# Install Python 3.11 if needed
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    apt update && apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Install system dependencies
echo "Installing system dependencies..."
apt install -y sox libsox-dev

# Install Poetry to persistent storage
if [ ! -d "/workspace/.poetry" ]; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/workspace/.poetry python3 -
fi
export PATH="/workspace/.poetry/bin:$PATH"

# Setup Poetry environment
cd /workspace/cosyvoice/tts-backend
poetry env use python3.11
poetry install

# Install CosyVoice dependencies
echo "Installing CosyVoice dependencies..."
poetry run pip install hyperpyyaml modelscope onnxruntime-gpu conformer WeTextProcessing \
    openai-whisper librosa soundfile inflect diffusers omegaconf hydra-core \
    transformers accelerate lightning pytorch-lightning rich gdown x-transformers \
    matplotlib pyworld --no-cache-dir

# Clone CosyVoice if not exists
if [ ! -d "/workspace/cosyvoice-official" ]; then
    echo "Cloning CosyVoice..."
    cd /workspace
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git cosyvoice-official
fi

# Initialize submodules
cd /workspace/cosyvoice-official
git submodule update --init --recursive

# Download model if not exists
if [ ! -d "/workspace/models/Fun-CosyVoice3-0.5B" ]; then
    echo "Downloading CosyVoice model..."
    cd /workspace/cosyvoice/tts-backend
    poetry run python -c "
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/workspace/models/Fun-CosyVoice3-0.5B')
"
fi

echo "=== Setup Complete ==="
echo "Run: cd /workspace/cosyvoice/tts-backend && make dev"
