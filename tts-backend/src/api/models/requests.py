"""API request models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.audio.formats import AudioFormat


class SynthesizeRequest(BaseModel):
    """Request model for speech synthesis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Text to synthesize into speech (1-5000 characters)",
    )
    voice_id: str = Field(
        default="default",
        description="ID of the voice to use for synthesis",
    )
    output_format: Literal["wav", "mp3", "ogg", "flac"] = Field(
        default="wav",
        description="Output audio format",
    )
    sample_rate: int = Field(
        default=22050,
        ge=8000,
        le=48000,
        description="Output sample rate in Hz",
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier",
    )
    language: str = Field(
        default="en",
        description="Language code (e.g., 'en', 'zh', 'ja')",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Clean and validate input text."""
        text = v.strip()
        if not text:
            raise ValueError("Text cannot be empty or whitespace only")
        return text

    model_config = {"extra": "ignore"}


class CloneVoiceRequest(BaseModel):
    """Request model for voice cloning."""

    voice_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Display name for the cloned voice",
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="Optional description of the voice",
    )
    language: str = Field(
        default="en",
        description="Primary language of the voice",
    )

    @field_validator("voice_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate voice name."""
        name = v.strip()
        if not name:
            raise ValueError("Voice name cannot be empty")
        return name

    model_config = {"extra": "ignore"}


# Aliases for backward compatibility
SynthesisRequest = SynthesizeRequest
StreamingSynthesisRequest = SynthesizeRequest
VoiceUploadRequest = CloneVoiceRequest
