"""CosyVoice TTS model implementation.

Exports:
- CosyVoiceConfig: Configuration for the engine
- CosyVoiceEngine: Main engine implementing TTSEngine
- CosyVoiceInference/MockCosyVoiceInference: Inference handlers
- TextPreprocessor: Text preprocessing
- VoiceManager: Voice profile management
"""

from src.models.cosyvoice.config import CosyVoiceConfig
from src.models.cosyvoice.engine import CosyVoiceEngine
from src.models.cosyvoice.inference import (
    CosyVoiceInference,
    MockCosyVoiceInference,
    get_inference,
)
from src.models.cosyvoice.preprocessor import (
    TextPreprocessor,
    ChineseTextPreprocessor,
    get_preprocessor,
)
from src.models.cosyvoice.voice_manager import VoiceManager, DEFAULT_VOICES

__all__ = [
    # Config
    "CosyVoiceConfig",
    # Engine
    "CosyVoiceEngine",
    # Inference
    "CosyVoiceInference",
    "MockCosyVoiceInference",
    "get_inference",
    # Preprocessor
    "TextPreprocessor",
    "ChineseTextPreprocessor",
    "get_preprocessor",
    # Voice Manager
    "VoiceManager",
    "DEFAULT_VOICES",
]
