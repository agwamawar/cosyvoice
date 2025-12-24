# RunPod Deployment Guide

This guide covers deploying the TTS Backend to RunPod with GPU support.

## Prerequisites

1. **RunPod Account**: Sign up at [runpod.io](https://runpod.io)
2. **Network Storage**: Create network storage for model weights (recommended 50GB+)
3. **API Credits**: Ensure sufficient credits for GPU pod rental

## 1. Upload Model Weights to Network Storage

### Create Network Storage

1. Go to RunPod Console → Storage → Network Volumes
2. Click "Create Network Volume"
3. Select a datacenter close to your target GPU pods
4. Allocate at least 50GB for CosyVoice model weights
5. Note the volume name (e.g., `tts-models`)

### Upload Model Files

Connect to a CPU pod with your network storage mounted:

```bash
# Clone CosyVoice repository
git clone https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice

# Download model weights (adjust as needed for your setup)
# The exact commands depend on the CosyVoice release
python download_models.py

# Copy to network storage
cp -r pretrained_models /runpod-volume/cosyvoice/
```

## 2. Create GPU Pod

### Recommended Pod Configuration

| Setting | Recommended Value |
|---------|-------------------|
| GPU | RTX 4090, A100, or similar |
| vCPU | 8+ cores |
| RAM | 32GB+ |
| Disk | 50GB+ |
| Network Storage | Mount your model storage |

### Pod Template Settings

1. Go to Pods → Deploy
2. Select GPU type (RTX 4090 recommended for cost/performance)
3. Under "Network Volume", attach your model storage
4. Use the official PyTorch image or our custom Docker image

## 3. Clone Repository on Pod

```bash
# SSH into your pod or use the web terminal
cd /workspace

# Clone the TTS backend
git clone https://github.com/your-username/tts-backend.git
cd tts-backend
```

## 4. Environment Configuration

Create your production `.env` file:

```bash
cat > .env << 'EOF'
# Production settings for RunPod
APP_NAME=tts-backend
APP_ENV=production
DEBUG=false

# Server
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
SERVER_WORKERS=4
SERVER_RELOAD=false

# CRITICAL: Disable mock mode for real inference
USE_MOCK=false

# Model path - adjust to your network storage mount
MODEL_PATH=/runpod-volume/cosyvoice
MODEL_DEVICE=cuda
MODEL_DTYPE=float16

# Audio
AUDIO_DEFAULT_SAMPLE_RATE=22050
AUDIO_DEFAULT_FORMAT=wav
AUDIO_MAX_DURATION=300

# API - USE STRONG KEYS IN PRODUCTION
API_KEYS=your-secure-api-key-here
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Voices
VOICES_DIRECTORY=/runpod-volume/voices
EOF
```

**Important Settings:**

| Variable | Production Value | Notes |
|----------|-----------------|-------|
| `USE_MOCK` | `false` | **Critical**: Enable real model inference |
| `MODEL_DEVICE` | `cuda` | Use GPU for inference |
| `MODEL_PATH` | `/runpod-volume/cosyvoice` | Path to model on network storage |
| `DEBUG` | `false` | Disable debug mode |
| `API_KEYS` | Strong random key | Generate with `openssl rand -hex 32` |

## 5. Install Dependencies

```bash
# Install Poetry if not available
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --only main

# Or use pip directly
pip install -r requirements.txt
```

## 6. Run the Server

### Direct Run

```bash
# Start the server
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Docker

```bash
# Build the production image
docker build -f docker/Dockerfile -t tts-backend .

# Run with GPU support
docker run --gpus all \
  -p 8000:8000 \
  -v /runpod-volume:/runpod-volume \
  --env-file .env \
  tts-backend
```

### As a Background Service

```bash
# Using nohup
nohup poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4 > server.log 2>&1 &

# Check logs
tail -f server.log
```

## 7. Test Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Readiness Check

```bash
curl http://localhost:8000/health/ready
```

Should show `model_loaded: true` when ready.

### Test Synthesis

```bash
curl -X POST http://localhost:8000/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text": "Hello from RunPod!", "voice_id": "default"}' \
  --output test.wav
```

## 8. Expose to Internet

### RunPod Proxy

RunPod automatically provides an HTTPS endpoint:

```
https://{POD_ID}-8000.proxy.runpod.net
```

### Custom Domain (Optional)

1. Set up a reverse proxy (nginx, Caddy)
2. Configure SSL certificates
3. Point your domain to the pod's public IP

## 9. Persistent Deployment

### Auto-start Script

Create `/workspace/start.sh`:

```bash
#!/bin/bash
cd /workspace/tts-backend
source .env
poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Supervisor

Install and configure supervisor for process management:

```bash
pip install supervisor

# Create supervisor config
cat > /etc/supervisor/conf.d/tts-backend.conf << 'EOF'
[program:tts-backend]
command=/root/.local/bin/poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
directory=/workspace/tts-backend
autostart=true
autorestart=true
stderr_logfile=/var/log/tts-backend.err.log
stdout_logfile=/var/log/tts-backend.out.log
EOF

# Start supervisor
supervisord -c /etc/supervisor/supervisord.conf
supervisorctl start tts-backend
```

## Troubleshooting

### Model Not Loading

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify model path
ls -la $MODEL_PATH

# Check logs
tail -f server.log
```

### Out of Memory

- Reduce `SERVER_WORKERS` to 1-2
- Use `MODEL_DTYPE=float16` for lower memory usage
- Consider a larger GPU

### Slow Response Times

- First request may be slow (model warmup)
- Check GPU utilization: `nvidia-smi`
- Consider using streaming endpoints for long text

## Cost Optimization

- Use spot instances for development
- Scale down workers during low traffic
- Use network storage to avoid re-downloading models
- Consider serverless options for variable workloads
