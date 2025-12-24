.PHONY: help install install-dev test lint format type-check clean run dev docker-build docker-up docker-down

# Default target
help:
	@echo "TTS Backend - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install all dependencies including dev"
	@echo "  setup          Full setup (install-dev + pre-commit)"
	@echo ""
	@echo "Development:"
	@echo "  run            Run the server in production mode"
	@echo "  dev            Run the server in development mode with reload"
	@echo "  test           Run tests with coverage"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-int       Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linter (ruff)"
	@echo "  format         Format code (ruff format)"
	@echo "  type-check     Run type checker (mypy)"
	@echo "  check          Run all checks (lint + type-check)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build   Build Docker image"
	@echo "  docker-up      Start services with docker-compose"
	@echo "  docker-down    Stop services"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Remove cache and build artifacts"
	@echo "  download-model Download CosyVoice model weights"

# Setup
install:
	poetry install --only main

install-dev:
	poetry install

setup: install-dev
	poetry run pre-commit install

# Development
run:
	poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000

dev:
	poetry run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Testing
test:
	poetry run pytest tests/unit tests/integration -v

test-unit:
	poetry run pytest tests/unit -v

test-integration:
	poetry run pytest tests/integration -v -m integration

test-fast:
	poetry run pytest tests/unit tests/integration -v -m "not slow"

test-coverage:
	poetry run pytest --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

# Alias for backwards compatibility
test-int: test-integration
test-cov: test-coverage

# Code Quality
lint:
	poetry run ruff check src tests

format:
	poetry run ruff format src tests
	poetry run ruff check --fix src tests

type-check:
	poetry run mypy src

check: lint type-check

# Docker
docker-build:
	docker build -f docker/Dockerfile -t tts-backend:latest .

docker-build-dev:
	docker build -f docker/Dockerfile.dev -t tts-backend:dev .

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

# Utilities
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

download-model:
	./scripts/download_model.sh
