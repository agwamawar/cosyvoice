"""TTS model implementations.

Registers available engines with the engine registry on import.
"""

from src.core.engine_registry import get_engine_registry
from src.models.base import BaseTTSModel
from src.models.f5tts.config import F5TTSConfig
from src.models.f5tts.engine import F5TTSEngine
from src.utils.logging import get_logger

logger = get_logger("system")

__all__ = [
    "BaseTTSModel",
    "F5TTSConfig",
    "F5TTSEngine",
    "register_engines",
]


def register_engines() -> None:
    """
    Register all available TTS engines with the registry.

    Call this function during application startup to make engines available.
    """
    registry = get_engine_registry()

    # Register F5-TTS engine as default
    config = F5TTSConfig()
    registry.register(
        name="f5tts",
        engine_class=F5TTSEngine,
        config={"config": config},
        set_default=True,
    )

    logger.info("Registered F5-TTS engine as default")


def get_f5tts_engine(use_mock: bool = False) -> F5TTSEngine:
    """
    Create an F5-TTS engine instance.

    Args:
        use_mock: Use mock inference for testing

    Returns:
        F5TTSEngine instance
    """
    config = F5TTSConfig(use_mock=use_mock)
    return F5TTSEngine(config)
