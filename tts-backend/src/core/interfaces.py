"""Abstract interfaces for TTS engines.

Defines:
- Pydantic models for synthesis options, results, voice info, etc.
- TTSEngine Protocol/ABC for engine implementations
- EngineCapabilities and HealthStatus models
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Literal

from pydantic import BaseModel, Field


# =============================================================================
# Synthesis Options and Results
# =============================================================================


class SynthesisOptions(BaseModel):
    """Options for speech synthesis."""

    voice_id: str = Field(default="default", description="Voice identifier to use")
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0)",
    )
    pitch: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Pitch adjustment (-1.0 to 1.0, 0.0 is neutral)",
    )
    emotion: str | None = Field(
        default=None,
        description="Emotion to apply (e.g., 'happy', 'sad', 'neutral')",
    )
    language: str = Field(default="en", description="Language code (e.g., 'en', 'zh')")
    output_format: str = Field(default="wav", description="Output audio format")
    sample_rate: int = Field(
        default=22050,
        ge=8000,
        le=48000,
        description="Output sample rate in Hz",
    )

    model_config = {"extra": "ignore"}


class AudioResult(BaseModel):
    """Result from speech synthesis."""

    audio_data: bytes = Field(..., description="Raw audio bytes")
    format: str = Field(..., description="Audio format (wav, mp3, etc.)")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    duration_seconds: float = Field(..., ge=0, description="Audio duration in seconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., voice_id, text_length)",
    )

    model_config = {"arbitrary_types_allowed": True}


class AudioSample(BaseModel):
    """Audio sample for voice cloning."""

    audio_data: bytes = Field(..., description="Raw audio bytes")
    format: str = Field(default="wav", description="Audio format")
    sample_rate: int = Field(default=22050, description="Sample rate in Hz")
    duration_seconds: float = Field(..., ge=0, description="Sample duration in seconds")

    model_config = {"arbitrary_types_allowed": True}


# =============================================================================
# Voice Information
# =============================================================================


class VoiceInfo(BaseModel):
    """Information about a TTS voice."""

    voice_id: str = Field(..., description="Unique voice identifier")
    name: str = Field(..., description="Display name for the voice")
    language: str = Field(default="en", description="Primary language code")
    gender: Literal["male", "female", "neutral"] | None = Field(
        default=None,
        description="Voice gender",
    )
    is_cloned: bool = Field(
        default=False,
        description="Whether this is a cloned/custom voice",
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the voice",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    preview_url: str | None = Field(
        default=None,
        description="URL to voice sample audio",
    )

    model_config = {"extra": "ignore"}


# =============================================================================
# Engine Capabilities and Health
# =============================================================================


class EngineCapabilities(BaseModel):
    """Capabilities of a TTS engine."""

    supports_cloning: bool = Field(
        default=False,
        description="Whether the engine supports voice cloning",
    )
    supports_streaming: bool = Field(
        default=False,
        description="Whether the engine supports streaming synthesis",
    )
    supports_emotions: bool = Field(
        default=False,
        description="Whether the engine supports emotion control",
    )
    supported_languages: list[str] = Field(
        default_factory=lambda: ["en"],
        description="List of supported language codes",
    )
    supported_formats: list[str] = Field(
        default_factory=lambda: ["wav"],
        description="List of supported output formats",
    )
    max_text_length: int = Field(
        default=5000,
        description="Maximum input text length in characters",
    )
    min_sample_rate: int = Field(default=8000, description="Minimum supported sample rate")
    max_sample_rate: int = Field(default=48000, description="Maximum supported sample rate")


class HealthStatus(BaseModel):
    """Health status of a TTS engine."""

    healthy: bool = Field(..., description="Whether the engine is healthy")
    latency_ms: float | None = Field(
        default=None,
        description="Last measured inference latency in milliseconds",
    )
    gpu_memory_used_mb: float | None = Field(
        default=None,
        description="GPU memory used in MB",
    )
    gpu_memory_total_mb: float | None = Field(
        default=None,
        description="Total GPU memory in MB",
    )
    model_loaded: bool = Field(default=False, description="Whether the model is loaded")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional health details",
    )


# =============================================================================
# TTS Engine Abstract Base Class
# =============================================================================


class TTSEngine(ABC):
    """
    Abstract base class for TTS engine implementations.

    All TTS engines must implement this interface to be compatible
    with the engine registry and TTS service.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name identifier."""
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the engine model is loaded and ready for inference."""
        ...

    @abstractmethod
    async def load(self) -> None:
        """
        Load the TTS model into memory.

        This should be called before any synthesis operations.
        May take significant time for large models.

        Raises:
            ModelLoadError: If model loading fails
        """
        ...

    @abstractmethod
    async def unload(self) -> None:
        """
        Unload the model and free resources.

        Should be called during shutdown or when switching engines.
        """
        ...

    @abstractmethod
    async def synthesize(self, text: str, options: SynthesisOptions) -> AudioResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            options: Synthesis options (voice, speed, pitch, etc.)

        Returns:
            AudioResult containing audio data and metadata

        Raises:
            ValidationError: If input validation fails
            InferenceError: If synthesis fails
            VoiceNotFoundError: If the specified voice doesn't exist
        """
        ...

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        options: SynthesisOptions,
        chunk_size_ms: int = 100,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized speech as audio chunks.

        Args:
            text: Text to synthesize
            options: Synthesis options
            chunk_size_ms: Size of each audio chunk in milliseconds

        Yields:
            Audio data chunks (first chunk may include header for WAV)

        Raises:
            ValidationError: If input validation fails
            InferenceError: If synthesis fails
        """
        ...

    @abstractmethod
    async def clone_voice(
        self,
        samples: list[AudioSample],
        voice_name: str,
        description: str | None = None,
    ) -> VoiceInfo:
        """
        Clone a voice from audio samples.

        Args:
            samples: List of audio samples for voice cloning
            voice_name: Name for the cloned voice
            description: Optional description

        Returns:
            VoiceInfo for the newly created voice

        Raises:
            ValidationError: If samples are invalid
            AudioProcessingError: If voice cloning fails
            NotImplementedError: If cloning is not supported
        """
        ...

    @abstractmethod
    async def get_available_voices(self) -> list[VoiceInfo]:
        """
        Get list of available voices.

        Returns:
            List of VoiceInfo for all available voices
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> EngineCapabilities:
        """
        Get engine capabilities.

        Returns:
            EngineCapabilities describing what this engine supports
        """
        ...

    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """
        Check engine health status.

        Returns:
            HealthStatus with health information
        """
        ...

    async def get_voice(self, voice_id: str) -> VoiceInfo | None:
        """
        Get information about a specific voice.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceInfo if found, None otherwise
        """
        voices = await self.get_available_voices()
        for voice in voices:
            if voice.voice_id == voice_id:
                return voice
        return None

    async def supports_voice(self, voice_id: str) -> bool:
        """
        Check if a voice ID is supported.

        Args:
            voice_id: Voice identifier

        Returns:
            True if the voice exists
        """
        voice = await self.get_voice(voice_id)
        return voice is not None


# Legacy alias for backward compatibility
TTSResult = AudioResult
