# TTS Backend Service

A high-performance Text-to-Speech backend service powered by **CosyVoice 3** and **FastAPI**, designed for production deployment on GPU instances.

[![CI](https://github.com/your-username/tts-backend/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/tts-backend/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Features

- **CosyVoice 3 Integration** - State-of-the-art neural TTS synthesis
- **Voice Cloning** - Clone voices from audio samples
- **Multiple Output Formats** - WAV, MP3, OGG, FLAC
- **Streaming Audio** - Real-time audio streaming support
- **FastAPI Backend** - High-performance async API with OpenAPI docs
- **Mock Mode** - Development without GPU requirements
- **Docker Ready** - Production-ready containerization
- **GPU Optimized** - CUDA support for fast inference

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tts-backend.git
cd tts-backend
```

### 2. Run the setup script

```bash
./scripts/setup.sh
```

Or manually:
```bash
poetry install
poetry run pre-commit install
cp .env.example .env
```

### 3. Start the development server

```bash
make dev
```

### 4. Open the API documentation

Visit **http://localhost:8000/docs** for the interactive Swagger UI.

## Development

### Available Make Commands

| Command | Description |
|---------|-------------|
| `make dev` | Start development server with hot reload |
| `make run` | Start production server |
| `make test` | Run all tests |
| `make test-unit` | Run unit tests only |
| `make test-integration` | Run integration tests |
| `make test-coverage` | Run tests with coverage report |
| `make lint` | Run ruff linter |
| `make format` | Format code with ruff |
| `make type-check` | Run mypy type checker |
| `make check` | Run all code quality checks |

### Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage report
make test-coverage

# Run specific test file
poetry run pytest tests/unit/test_audio/test_validation.py -v
```

### Code Quality

```bash
# Run all checks before committing
make check

# Or run pre-commit on all files
poetry run pre-commit run --all-files
```

## API Overview

### Health Endpoints (No Auth Required)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health status |
| `/health/ready` | GET | Readiness probe (model loaded) |
| `/health/live` | GET | Liveness probe |

### Voice Endpoints (Auth Required)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/voices` | GET | List available voices |
| `/v1/voices/{voice_id}` | GET | Get voice details |
| `/v1/voices/clone` | POST | Clone voice from audio |
| `/v1/voices/{voice_id}` | DELETE | Delete cloned voice |

### Synthesis Endpoints (Auth Required)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/tts/synthesize` | POST | Synthesize speech (file download) |
| `/v1/tts/synthesize/json` | POST | Synthesize speech (base64 JSON) |
| `/v1/tts/synthesize/stream` | POST | Streaming synthesis |

### Authentication

Protected endpoints require the `X-API-Key` header:
```bash
curl -X POST http://localhost:8000/v1/tts/synthesize \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice_id": "default"}'
```

## Configuration

All configuration is done via environment variables. See [`.env.example`](.env.example) for all available options.

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (dev/staging/prod) | `dev` |
| `DEBUG` | Enable debug mode | `false` |
| `MODEL_USE_MOCK` | Use mock engine (no GPU) | `true` |
| `MODEL_PATH` | Path to CosyVoice model | `./models/cosyvoice` |
| `MODEL_DEVICE` | Device (cuda/cpu/auto) | `auto` |
| `API_KEYS` | Comma-separated API keys | `` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Deployment

### Docker (Development)

```bash
# Build and start
make docker-build-dev
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

### Docker (Production)

```bash
# Build production image
make docker-build

# Run with GPU support
docker run --gpus all -p 8000:8000 \
  -e MODEL_USE_MOCK=false \
  -e API_KEYS=your-secret-key \
  tts-backend:latest
```

### RunPod Deployment

1. Build and push the Docker image to a registry
2. Create a new RunPod GPU pod (recommended: RTX 4090 or A100)
3. Configure environment variables:
   - `MODEL_USE_MOCK=false`
   - `MODEL_DEVICE=cuda`
   - `API_KEYS=your-production-key`
4. Mount the models volume at `/app/models`
5. Expose port 8000

See [`docker/Dockerfile`](docker/Dockerfile) for production build details.

## Project Structure

```
tts-backend/
├── src/
│   ├── api/           # FastAPI routers, models, middleware
│   ├── audio/         # Audio processing and conversion
│   ├── config/        # Settings and configuration
│   ├── core/          # TTS service interfaces
│   └── models/        # CosyVoice engine implementation
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # Integration tests
├── docker/            # Docker configuration
├── scripts/           # Utility scripts
├── voices/            # Voice profile storage
└── .github/           # CI/CD workflows
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make check && make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request
