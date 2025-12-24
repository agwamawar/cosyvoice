# TTS Backend Service

A high-performance Text-to-Speech backend service powered by CosyVoice 3 and FastAPI. Designed for easy local development with mock mode and production deployment on GPU instances like RunPod.

## Features

- **CosyVoice 3 Integration**: State-of-the-art neural TTS synthesis
- **FastAPI Backend**: High-performance async API with OpenAPI docs
- **Multiple Output Formats**: WAV, MP3, OGG, FLAC support
- **Voice Cloning**: Clone voices from audio samples
- **Streaming Audio**: Real-time chunked audio streaming
- **Mock Mode**: Full development workflow without GPU
- **Docker Ready**: Production and development Dockerfiles
- **CI/CD**: GitHub Actions for testing and Docker builds

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/tts-backend.git
cd tts-backend
```

### 2. Run Setup Script

```bash
./scripts/setup.sh
```

### 3. Start Development Server

```bash
make dev
```

### 4. Open API Documentation

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API docs.

## Development

### Make Commands

```bash
make install       # Install production dependencies
make dev           # Run dev server with hot reload
make test          # Run all tests
make test-unit     # Run unit tests only
make test-coverage # Run tests with coverage report
make lint          # Run ruff linter
make format        # Format code with ruff
make type-check    # Run mypy type checker
make clean         # Remove cache files
```

### Testing

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests
make test-integration

# Generate coverage report
make test-coverage
# Coverage report available at htmlcov/index.html
```

### Linting & Formatting

```bash
# Check for issues
make lint

# Auto-fix and format
make format

# Type checking
make type-check
```

## Deployment

### Docker

```bash
# Build production image
make docker-build

# Build development image
make docker-build-dev

# Run with docker-compose
docker-compose -f docker/docker-compose.yml up
```

### RunPod GPU Deployment

For detailed RunPod deployment instructions, see [docs/RUNPOD.md](docs/RUNPOD.md).

Quick overview:
1. Upload model weights to RunPod network storage
2. Deploy with GPU pod template
3. Set `USE_MOCK=false` and `MODEL_DEVICE=cuda`
4. Run the server

## API Overview

All endpoints except health checks require `X-API-Key` header.

### Health Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /health` | Service health status |
| `GET /health/ready` | Readiness probe (model loaded) |
| `GET /health/live` | Liveness probe |

### TTS Endpoints
| Endpoint | Description |
|----------|-------------|
| `POST /v1/tts/synthesize` | Synthesize speech (returns audio file) |
| `POST /v1/tts/synthesize/json` | Synthesize speech (returns base64 JSON) |
| `POST /v1/tts/synthesize/stream` | Streaming synthesis |

### Voice Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /v1/voices` | List available voices |
| `GET /v1/voices/{voice_id}` | Get voice details |
| `POST /v1/voices/clone` | Clone a voice from audio |
| `DELETE /v1/voices/{voice_id}` | Delete a cloned voice |

For full API documentation, see [docs/API.md](docs/API.md).

## Configuration

All configuration is done via environment variables. Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### Key Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_MOCK` | Mock mode (no GPU required) | `true` |
| `MODEL_PATH` | Path to CosyVoice model | `./models/cosyvoice` |
| `MODEL_DEVICE` | Inference device (cuda/cpu/auto) | `auto` |
| `API_KEYS` | Comma-separated API keys | `dev-key-123` |
| `VOICES_DIRECTORY` | Voice profiles directory | `./voices` |

See [.env.example](.env.example) for all available options.

## Project Structure

```
tts-backend/
├── src/
│   ├── api/           # FastAPI routers and models
│   ├── audio/         # Audio processing and conversion
│   ├── config/        # Settings and configuration
│   ├── core/          # TTS service and interfaces
│   ├── models/        # CosyVoice engine implementation
│   └── utils/         # Logging and utilities
├── tests/
│   ├── unit/          # Unit tests
│   └── integration/   # API integration tests
├── docker/            # Dockerfile and compose files
├── scripts/           # Setup and utility scripts
├── docs/              # Documentation
├── voices/            # Voice profile storage
└── models/            # Model weights (gitignored)
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make test && make lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request
