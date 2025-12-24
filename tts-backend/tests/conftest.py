"""Pytest configuration and fixtures.

Global fixtures for all tests:
- settings: test settings with USE_MOCK=true
- app: FastAPI test application
- client: httpx.AsyncClient for testing
- auth_headers: dict with valid X-API-Key
- mock_engine: configured mock TTS engine
- tts_service: service with mock engine
- audio_processor: real audio processor
- sample_wav_bytes: valid WAV audio (silence, 3 seconds)
- sample_text: example text for synthesis
"""

import asyncio
import io
import os
import struct
import wave
from collections.abc import AsyncGenerator, Generator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.app import create_app
from src.audio.processor import AudioProcessor
from src.config import Settings, get_settings
from src.core.engine_registry import EngineRegistry
from src.core.tts_service import TTSService
from src.models.cosyvoice import CosyVoiceConfig, CosyVoiceEngine


# -----------------------------------------------------------------------------
# Event Loop
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# -----------------------------------------------------------------------------
# Settings Fixture
# -----------------------------------------------------------------------------
@pytest.fixture
def settings() -> Settings:
    """Create test settings with mock mode enabled."""
    # Set environment for testing
    os.environ["ENVIRONMENT"] = "dev"
    os.environ["DEBUG"] = "true"
    os.environ["USE_MOCK"] = "true"
    os.environ["COSYVOICE_USE_MOCK"] = "true"
    os.environ["API_AUTH_ENABLED"] = "true"
    os.environ["API_KEYS"] = "test-api-key-123"
    os.environ["VOICES_DIRECTORY"] = "./test_voices"

    # Clear cached settings
    get_settings.cache_clear()

    return get_settings()


# -----------------------------------------------------------------------------
# App and Client Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
async def app(
    settings: Settings, registry_with_mock: EngineRegistry
) -> AsyncGenerator[FastAPI, None]:
    """Create test FastAPI application with initialized TTS service."""
    application = create_app(settings)

    # Initialize the TTS service for testing (simulating lifespan startup)
    tts_service = TTSService(settings, registry=registry_with_mock)
    await tts_service.initialize()
    application.state.tts_service = tts_service

    yield application

    # Cleanup
    await tts_service.shutdown()


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# -----------------------------------------------------------------------------
# Auth Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers with valid API key for authenticated requests."""
    return {
        "Content-Type": "application/json",
        "X-API-Key": "test-api-key-123",
    }


@pytest.fixture
def no_auth_headers() -> dict[str, str]:
    """Headers without API key."""
    return {
        "Content-Type": "application/json",
    }


# -----------------------------------------------------------------------------
# Engine Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
async def mock_engine() -> AsyncGenerator[CosyVoiceEngine, None]:
    """Create configured mock TTS engine."""
    config = CosyVoiceConfig(use_mock=True)
    engine = CosyVoiceEngine(config)
    await engine.load()
    yield engine
    await engine.unload()


@pytest.fixture
def registry_with_mock() -> EngineRegistry:
    """Get engine registry and register mock engine."""
    registry = EngineRegistry()
    # Register the mock CosyVoice engine with proper config object
    mock_config = CosyVoiceConfig(use_mock=True)
    registry.register(
        "cosyvoice",
        CosyVoiceEngine,
        config={"config": mock_config},
        set_default=True,
    )
    return registry


# -----------------------------------------------------------------------------
# Service Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
async def tts_service(
    settings: Settings, registry_with_mock: EngineRegistry
) -> AsyncGenerator[TTSService, None]:
    """Create TTS service with mock engine."""
    service = TTSService(settings, registry=registry_with_mock)
    await service.initialize()
    yield service
    await service.shutdown()


# -----------------------------------------------------------------------------
# Audio Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def audio_processor() -> AudioProcessor:
    """Create real audio processor."""
    return AudioProcessor()


@pytest.fixture
def sample_wav_bytes() -> bytes:
    """Generate valid WAV audio (silence, 3 seconds)."""
    sample_rate = 22050
    duration = 3.0
    num_samples = int(sample_rate * duration)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))

    return buffer.getvalue()


@pytest.fixture
def short_wav_bytes() -> bytes:
    """Generate short WAV audio (0.5 seconds) - below minimum duration."""
    sample_rate = 22050
    duration = 0.5
    num_samples = int(sample_rate * duration)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))

    return buffer.getvalue()


@pytest.fixture
def corrupted_audio_bytes() -> bytes:
    """Generate corrupted audio data."""
    return b"not valid audio data - corrupted"


# -----------------------------------------------------------------------------
# Text Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_text() -> str:
    """Example text for synthesis tests."""
    return "Hello, this is a test of the text to speech system."


@pytest.fixture
def sample_text_long() -> str:
    """Longer sample text for testing."""
    return (
        "The quick brown fox jumps over the lazy dog. "
        "This is a longer text sample that can be used to test "
        "the text-to-speech synthesis with multiple sentences. "
        "It includes various punctuation marks, and numbers like 123."
    )


# -----------------------------------------------------------------------------
# Request Data Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def synthesis_request_data() -> dict[str, Any]:
    """Sample synthesis request data."""
    return {
        "text": "Hello, world!",
        "voice_id": "default",
        "language": "en",
        "output_format": "wav",
        "sample_rate": 22050,
        "speed": 1.0,
    }


@pytest.fixture
def voice_upload_data() -> dict[str, Any]:
    """Sample voice upload/clone data."""
    return {
        "voice_name": "Test Voice",
        "description": "A test voice for unit testing",
    }
