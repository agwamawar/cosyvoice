"""Core TTS service components.

Exports:
- Interfaces: TTSEngine, SynthesisOptions, AudioResult, VoiceInfo, etc.
- Registry: EngineRegistry, get_engine_registry
- Models: SynthesisRequest, SynthesisResponse, CloneVoiceRequest, etc.
- Service: TTSService
"""

from src.core.engine_registry import (
    EngineRegistration,
    EngineRegistry,
    get_engine_registry,
)
from src.core.interfaces import (
    AudioResult,
    AudioSample,
    EngineCapabilities,
    HealthStatus,
    SynthesisOptions,
    TTSEngine,
    TTSResult,
    VoiceInfo,
)
from src.core.models import (
    AudioSampleMetadata,
    CloneVoiceRequest,
    EngineHealthResponse,
    EngineStatus,
    ServiceHealthResponse,
    StreamingSynthesisRequest,
    SynthesisRequest,
    SynthesisResponse,
    SynthesisTask,
    Voice,
    VoiceInfoResponse,
    VoiceListResponse,
)
from src.core.tts_service import TTSService

__all__ = [
    # Interfaces
    "TTSEngine",
    "SynthesisOptions",
    "AudioResult",
    "AudioSample",
    "VoiceInfo",
    "EngineCapabilities",
    "HealthStatus",
    "TTSResult",
    # Registry
    "EngineRegistry",
    "EngineRegistration",
    "get_engine_registry",
    # Models
    "SynthesisRequest",
    "SynthesisResponse",
    "StreamingSynthesisRequest",
    "CloneVoiceRequest",
    "AudioSampleMetadata",
    "VoiceListResponse",
    "VoiceInfoResponse",
    "ServiceHealthResponse",
    "EngineHealthResponse",
    "Voice",
    "SynthesisTask",
    "EngineStatus",
    # Service
    "TTSService",
]
