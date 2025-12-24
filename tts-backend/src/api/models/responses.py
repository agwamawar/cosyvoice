"""API response models."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Error detail model."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional error details")
    request_id: str | None = Field(default=None, description="Request ID for tracing")


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy", "degraded"] = Field(
        ..., description="Overall health status"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Application version")


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    ready: bool = Field(..., description="Whether the service is ready")
    checks: dict[str, bool] = Field(default_factory=dict)


class VoiceResponse(BaseModel):
    """Voice details response."""

    voice_id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Display name")
    language: str = Field(..., description="Primary language")
    gender: Literal["male", "female", "neutral"] | None = Field(default=None)
    is_cloned: bool = Field(default=False, description="Whether this is a cloned voice")
    description: str | None = Field(default=None, description="Voice description")


class VoiceListResponse(BaseModel):
    """List of voices response."""

    voices: list[VoiceResponse] = Field(default_factory=list)
    total: int = Field(..., description="Total number of voices")


class SynthesizeResponse(BaseModel):
    """Speech synthesis JSON response."""

    audio_base64: str = Field(..., description="Base64 encoded audio data")
    format: str = Field(..., description="Audio format")
    sample_rate: int = Field(..., description="Sample rate of the audio")
    duration_seconds: float = Field(..., description="Duration of synthesized audio in seconds")
    request_id: str = Field(..., description="Unique request identifier")
    voice_id: str = Field(..., description="Voice used for synthesis")
    text_length: int = Field(..., description="Number of characters synthesized")
    processing_time_ms: float = Field(..., description="Time taken to synthesize in milliseconds")


# Aliases for backward compatibility
SynthesisResponse = SynthesizeResponse
