"""TTS model implementations.

Registers available engines with the engine registry on import.
"""

from src.core.engine_registry import get_engine_registry
from src.models.base import BaseTTSModel
from src.models.cosyvoice.config import CosyVoiceConfig
from src.models.cosyvoice.engine import CosyVoiceEngine
from src.utils.logging import get_logger

logger = get_logger("system")

__all__ = [
    "BaseTTSModel",
    "CosyVoiceConfig",
    "CosyVoiceEngine",
    "register_engines",
]


def register_engines() -> None:
    """
    Register all available TTS engines with the registry.

    Call this function during application startup to make engines available.
    """
    registry = get_engine_registry()

    # Register CosyVoice engine
    config = CosyVoiceConfig()
    registry.register(
        name="cosyvoice",
        engine_class=CosyVoiceEngine,
        config={"config": config},
        set_default=True,
    )

    logger.info("Registered CosyVoice engine as default")


def get_cosyvoice_engine(use_mock: bool = False) -> CosyVoiceEngine:
    """
    Create a CosyVoice engine instance.

    Args:
        use_mock: Use mock inference for testing

    Returns:
        CosyVoiceEngine instance
    """
    config = CosyVoiceConfig(use_mock=use_mock)
    return CosyVoiceEngine(config)
