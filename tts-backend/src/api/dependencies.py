"""FastAPI dependencies for dependency injection."""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, Request

from src.api.exceptions import InvalidAPIKeyError, MissingAPIKeyError, ModelNotLoadedError
from src.audio.processor import AudioProcessor
from src.config import Settings, get_settings
from src.core.engine_registry import EngineRegistry, get_engine_registry
from src.core.tts_service import TTSService


@lru_cache
def get_settings_cached() -> Settings:
    """Get settings (cached)."""
    return get_settings()


def get_registry() -> EngineRegistry:
    """Get the engine registry (singleton)."""
    return get_engine_registry()


@lru_cache
def get_audio_processor() -> AudioProcessor:
    """Get the audio processor (cached)."""
    return AudioProcessor()


def get_tts_service(request: Request) -> TTSService:
    """Get the TTS service from application state."""
    service: TTSService = request.app.state.tts_service
    if not service.is_ready():
        raise ModelNotLoadedError()
    return service


def get_request_id(request: Request) -> str:
    """Get the request ID from request state."""
    return getattr(request.state, "request_id", "unknown")


async def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings_cached),
) -> str:
    """
    Verify API key for protected routes.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        Validated API key

    Raises:
        MissingAPIKeyError: If no API key provided
        InvalidAPIKeyError: If API key is invalid
    """
    if not settings.api.auth_enabled:
        return "auth-disabled"

    if not x_api_key:
        raise MissingAPIKeyError()

    if x_api_key not in settings.api.keys:
        raise InvalidAPIKeyError()

    return x_api_key


# Type aliases for cleaner dependency injection
SettingsDep = Annotated[Settings, Depends(get_settings_cached)]
RegistryDep = Annotated[EngineRegistry, Depends(get_registry)]
AudioProcessorDep = Annotated[AudioProcessor, Depends(get_audio_processor)]
TTSServiceDep = Annotated[TTSService, Depends(get_tts_service)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
APIKeyDep = Annotated[str, Depends(verify_api_key)]

# Dependency for requiring API key on routes
require_api_key = Depends(verify_api_key)
