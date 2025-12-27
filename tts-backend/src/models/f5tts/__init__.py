"""F5-TTS model implementation.

Provides:
- F5TTSEngine: TTS engine using F5-TTS for voice cloning
- F5TTSConfig: Configuration for F5-TTS
- F5TTSInference: Inference handler wrapping F5-TTS API
"""

from src.models.f5tts.config import F5TTSConfig
from src.models.f5tts.engine import F5TTSEngine

__all__ = [
    "F5TTSConfig",
    "F5TTSEngine",
]
