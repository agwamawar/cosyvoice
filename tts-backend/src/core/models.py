"""Service layer request/response models.

Defines Pydantic models for:
- SynthesisRequest/Response
- CloneVoiceRequest
- VoiceListResponse
- And other service-layer data structures
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Synthesis Request/Response
# =============================================================================


class SynthesisRequest(BaseModel):
    """Request for speech synthesis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to synthesize",
    )
    voice_id: str = Field(
        default="default",
        description="Voice identifier to use",
    )
    # Flattened SynthesisOptions fields
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier",
    )
    pitch: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Pitch adjustment (-1.0 to 1.0)",
    )
    emotion: str | None = Field(
        default=None,
        description="Emotion to apply",
    )
    language: str = Field(
        default="en",
        description="Language code",
    )
    output_format: str = Field(
        default="wav",
        description="Output audio format",
    )
    sample_rate: int = Field(
        default=22050,
        ge=8000,
        le=48000,
        description="Output sample rate",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Strip and validate text."""
        text = v.strip()
        if not text:
            raise ValueError("Text cannot be empty or whitespace only")
        return text

    model_config = {"extra": "ignore"}


class SynthesisResponse(BaseModel):
    """Response from speech synthesis."""

    audio_base64: str = Field(..., description="Base64 encoded audio data")
    format: str = Field(..., description="Audio format (wav, mp3, etc.)")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    request_id: str = Field(..., description="Unique request identifier")
    voice_id: str = Field(..., description="Voice used for synthesis")
    text_length: int = Field(..., description="Length of input text")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    model_config = {"extra": "ignore"}


class StreamingSynthesisRequest(SynthesisRequest):
    """Request for streaming speech synthesis."""

    chunk_size_ms: int = Field(
        default=100,
        ge=50,
        le=1000,
        description="Size of each audio chunk in milliseconds",
    )


# =============================================================================
# Voice Cloning Request/Response
# =============================================================================


class AudioSampleMetadata(BaseModel):
    """Metadata for an audio sample used in voice cloning."""

    filename: str = Field(..., description="Original filename")
    format: str = Field(default="wav", description="Audio format")
    duration_seconds: float | None = Field(
        default=None,
        description="Duration in seconds",
    )
    sample_rate: int | None = Field(
        default=None,
        description="Sample rate in Hz",
    )


class CloneVoiceRequest(BaseModel):
    """Request for voice cloning."""

    voice_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Name for the cloned voice",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Description of the voice",
    )
    language: str = Field(
        default="en",
        description="Primary language of the voice",
    )
    samples: list[AudioSampleMetadata] = Field(
        default_factory=list,
        description="Metadata for audio samples",
    )

    @field_validator("voice_name")
    @classmethod
    def validate_voice_name(cls, v: str) -> str:
        """Strip and validate voice name."""
        name = v.strip()
        if not name:
            raise ValueError("Voice name cannot be empty")
        return name

    model_config = {"extra": "ignore"}


# =============================================================================
# Voice List Response
# =============================================================================


class VoiceListResponse(BaseModel):
    """Response containing list of voices."""

    voices: list["VoiceInfoResponse"] = Field(
        default_factory=list,
        description="List of available voices",
    )
    total: int = Field(..., description="Total number of voices")

    model_config = {"extra": "ignore"}


class VoiceInfoResponse(BaseModel):
    """Voice information in API response format."""

    voice_id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Display name")
    language: str = Field(..., description="Primary language")
    gender: Literal["male", "female", "neutral"] | None = Field(
        default=None,
        description="Voice gender",
    )
    is_cloned: bool = Field(
        default=False,
        description="Whether this is a cloned voice",
    )
    description: str | None = Field(
        default=None,
        description="Voice description",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    preview_url: str | None = Field(
        default=None,
        description="URL to preview audio",
    )
    created_at: datetime | None = Field(
        default=None,
        description="Creation timestamp",
    )

    model_config = {"extra": "ignore"}


# =============================================================================
# Health and Status Models
# =============================================================================


class ServiceHealthResponse(BaseModel):
    """Overall service health response."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall health status",
    )
    version: str = Field(..., description="Service version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    engines: dict[str, "EngineHealthResponse"] = Field(
        default_factory=dict,
        description="Health status of each engine",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp",
    )


class EngineHealthResponse(BaseModel):
    """Health response for a single engine."""

    name: str = Field(..., description="Engine name")
    healthy: bool = Field(..., description="Whether engine is healthy")
    loaded: bool = Field(..., description="Whether model is loaded")
    latency_ms: float | None = Field(
        default=None,
        description="Last inference latency",
    )
    gpu_memory_used_mb: float | None = Field(
        default=None,
        description="GPU memory used",
    )
    gpu_memory_total_mb: float | None = Field(
        default=None,
        description="Total GPU memory",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details",
    )


# =============================================================================
# Legacy Dataclass Models (for backward compatibility)
# =============================================================================


class Voice(BaseModel):
    """Voice model (Pydantic version for consistency)."""

    id: str = Field(..., alias="voice_id")
    name: str
    language: str = "en"
    description: str | None = None
    gender: Literal["male", "female", "neutral"] | None = None
    tags: list[str] = Field(default_factory=list)
    is_custom: bool = False
    sample_url: str | None = None
    created_at: datetime | None = None
    reference_audio_path: str | None = None

    model_config = {"populate_by_name": True, "extra": "ignore"}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "description": self.description,
            "gender": self.gender,
            "tags": self.tags,
            "is_custom": self.is_custom,
            "sample_url": self.sample_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SynthesisTask(BaseModel):
    """Task representing a synthesis request."""

    task_id: str
    text: str
    voice_id: str
    language: str = "en"
    output_format: str = "wav"
    sample_rate: int = 22050
    speed: float = 1.0
    pitch: float = 0.0
    status: Literal["pending", "processing", "completed", "failed"] = "pending"
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    model_config = {"extra": "ignore"}


class EngineStatus(BaseModel):
    """Status of a TTS engine."""

    name: str
    is_loaded: bool
    device: str = "cpu"
    model_path: str | None = None
    memory_usage_mb: float | None = None
    supported_languages: list[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


# Update forward references
VoiceListResponse.model_rebuild()
