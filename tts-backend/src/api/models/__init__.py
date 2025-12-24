"""API request and response models."""

from src.api.models.requests import (
    CloneVoiceRequest,
    SynthesisRequest,
    SynthesizeRequest,
    StreamingSynthesisRequest,
    VoiceUploadRequest,
)
from src.api.models.responses import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    SynthesisResponse,
    SynthesizeResponse,
    VoiceListResponse,
    VoiceResponse,
)

__all__ = [
    # Requests
    "SynthesizeRequest",
    "CloneVoiceRequest",
    "SynthesisRequest",
    "StreamingSynthesisRequest",
    "VoiceUploadRequest",
    # Responses
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ReadinessResponse",
    "SynthesizeResponse",
    "SynthesisResponse",
    "VoiceListResponse",
    "VoiceResponse",
]
