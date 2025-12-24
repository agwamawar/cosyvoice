"""TTS Service - High-level orchestration of TTS operations.

Provides:
- TTSService class for synthesis, voice management, health checks
- Input validation and error translation
- Logging with timing for all operations
"""

import base64
import time
import uuid
from typing import AsyncGenerator

from src.api.exceptions import (
    AudioProcessingError,
    InferenceError,
    ModelNotLoadedError,
    TextTooLongError,
    ValidationError,
    VoiceNotFoundError,
)
from src.core.engine_registry import EngineRegistry, get_engine_registry
from src.core.interfaces import (
    AudioResult,
    AudioSample,
    EngineCapabilities,
    HealthStatus,
    SynthesisOptions,
    TTSEngine,
    VoiceInfo,
)
from src.core.models import (
    CloneVoiceRequest,
    EngineHealthResponse,
    ServiceHealthResponse,
    SynthesisRequest,
    SynthesisResponse,
    VoiceInfoResponse,
    VoiceListResponse,
)
from src.utils.logging import get_logger
from src.utils.timing import Timer

logger = get_logger("inference")


# MIME types for audio formats
AUDIO_MIME_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


class TTSService:
    """
    High-level TTS service that orchestrates synthesis operations.

    This service:
    - Validates inputs
    - Gets engine from registry
    - Calls engine methods
    - Handles errors and translates to custom exceptions
    - Logs operations with timing
    """

    def __init__(
        self,
        registry: EngineRegistry | None = None,
        audio_processor: "AudioProcessor | None" = None,
    ) -> None:
        """
        Initialize the TTS service.

        Args:
            registry: Engine registry (uses global registry if None)
            audio_processor: Audio processor for format conversion (optional)
        """
        self._registry = registry or get_engine_registry()
        self._audio_processor = audio_processor
        self._start_time = time.time()
        self._default_engine_name: str | None = None

    @property
    def uptime_seconds(self) -> float:
        """Get service uptime in seconds."""
        return time.time() - self._start_time

    async def _get_engine(self, engine_name: str | None = None) -> TTSEngine:
        """
        Get a TTS engine, ensuring it's loaded.

        Args:
            engine_name: Specific engine name (uses default if None)

        Returns:
            Loaded TTSEngine instance

        Raises:
            ModelNotLoadedError: If no engine is available
        """
        try:
            engine = await self._registry.get_engine(engine_name)
            if not engine.is_loaded:
                raise ModelNotLoadedError()
            return engine
        except ValueError as e:
            logger.error(f"Failed to get engine: {e}")
            raise ModelNotLoadedError() from e

    def _validate_text(self, text: str, max_length: int = 10000) -> str:
        """Validate and clean input text."""
        if not text:
            raise ValidationError("Text cannot be empty")

        text = text.strip()
        if not text:
            raise ValidationError("Text cannot be empty or whitespace only")

        if len(text) > max_length:
            raise TextTooLongError(max_length=max_length, actual_length=len(text))

        return text

    def _validate_voice_id(self, voice_id: str) -> str:
        """Validate voice ID format."""
        if not voice_id:
            return "default"
        return voice_id.strip()

    def _build_synthesis_options(self, request: SynthesisRequest) -> SynthesisOptions:
        """Build SynthesisOptions from a SynthesisRequest."""
        return SynthesisOptions(
            voice_id=request.voice_id,
            speed=request.speed,
            pitch=request.pitch,
            emotion=request.emotion,
            language=request.language,
            output_format=request.output_format,
            sample_rate=request.sample_rate,
        )

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResponse:
        """
        Synthesize speech from a SynthesisRequest.

        Args:
            request: Synthesis request with text and options

        Returns:
            SynthesisResponse with base64 audio and metadata

        Raises:
            ValidationError: If input validation fails
            VoiceNotFoundError: If voice doesn't exist
            InferenceError: If synthesis fails
            ModelNotLoadedError: If engine not loaded
        """
        request_id = str(uuid.uuid4())

        with Timer("synthesize", log=True, log_level="info") as timer:
            # Validate inputs
            text = self._validate_text(request.text)
            voice_id = self._validate_voice_id(request.voice_id)

            logger.info(
                "Starting synthesis",
                request_id=request_id,
                text_length=len(text),
                voice_id=voice_id,
                language=request.language,
            )

            # Get engine
            engine = await self._get_engine()

            # Check voice exists
            voice = await engine.get_voice(voice_id)
            if voice is None:
                raise VoiceNotFoundError(voice_id)

            # Build options
            options = self._build_synthesis_options(request)
            options.voice_id = voice_id

            # Perform synthesis
            try:
                result = await engine.synthesize(text, options)
            except Exception as e:
                logger.error(
                    "Synthesis failed",
                    request_id=request_id,
                    error=str(e),
                )
                raise InferenceError(
                    message="Speech synthesis failed",
                    reason=str(e),
                ) from e

            # Encode audio as base64
            audio_base64 = base64.b64encode(result.audio_data).decode("utf-8")

            logger.info(
                "Synthesis completed",
                request_id=request_id,
                duration_seconds=result.duration_seconds,
                audio_size_bytes=len(result.audio_data),
            )

        return SynthesisResponse(
            audio_base64=audio_base64,
            format=result.format,
            sample_rate=result.sample_rate,
            duration_seconds=result.duration_seconds,
            request_id=request_id,
            voice_id=voice_id,
            text_length=len(text),
            processing_time_ms=timer.elapsed_ms,
        )

    async def synthesize_to_bytes(
        self,
        request: SynthesisRequest,
    ) -> tuple[bytes, str]:
        """
        Synthesize speech and return raw bytes with content type.

        Args:
            request: Synthesis request

        Returns:
            Tuple of (audio_bytes, content_type)

        Raises:
            ValidationError: If input validation fails
            VoiceNotFoundError: If voice doesn't exist
            InferenceError: If synthesis fails
        """
        request_id = str(uuid.uuid4())

        with Timer("synthesize_to_bytes", log=True, log_level="info"):
            # Validate inputs
            text = self._validate_text(request.text)
            voice_id = self._validate_voice_id(request.voice_id)

            logger.info(
                "Starting synthesis (bytes)",
                request_id=request_id,
                text_length=len(text),
                voice_id=voice_id,
            )

            # Get engine
            engine = await self._get_engine()

            # Check voice exists
            voice = await engine.get_voice(voice_id)
            if voice is None:
                raise VoiceNotFoundError(voice_id)

            # Build options
            options = self._build_synthesis_options(request)
            options.voice_id = voice_id

            # Perform synthesis
            try:
                result = await engine.synthesize(text, options)
            except Exception as e:
                logger.error("Synthesis failed", request_id=request_id, error=str(e))
                raise InferenceError(message="Speech synthesis failed", reason=str(e)) from e

            # Get content type
            content_type = AUDIO_MIME_TYPES.get(result.format, "audio/wav")

            logger.info(
                "Synthesis completed (bytes)",
                request_id=request_id,
                duration_seconds=result.duration_seconds,
                audio_size_bytes=len(result.audio_data),
            )

        return result.audio_data, content_type

    async def synthesize_stream(
        self,
        request: SynthesisRequest,
        chunk_size_ms: int = 100,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized speech as audio chunks.

        Args:
            request: Synthesis request
            chunk_size_ms: Size of each chunk in milliseconds

        Yields:
            Audio data chunks

        Raises:
            ValidationError: If input validation fails
            VoiceNotFoundError: If voice doesn't exist
            InferenceError: If synthesis fails
        """
        request_id = str(uuid.uuid4())

        # Validate inputs
        text = self._validate_text(request.text)
        voice_id = self._validate_voice_id(request.voice_id)

        logger.info(
            "Starting streaming synthesis",
            request_id=request_id,
            text_length=len(text),
            voice_id=voice_id,
            chunk_size_ms=chunk_size_ms,
        )

        # Get engine
        engine = await self._get_engine()

        # Check voice exists
        voice = await engine.get_voice(voice_id)
        if voice is None:
            raise VoiceNotFoundError(voice_id)

        # Build options
        options = self._build_synthesis_options(request)
        options.voice_id = voice_id

        # Stream synthesis
        try:
            async for chunk in engine.synthesize_stream(text, options, chunk_size_ms):
                yield chunk
        except Exception as e:
            logger.error("Streaming synthesis failed", request_id=request_id, error=str(e))
            raise InferenceError(message="Streaming synthesis failed", reason=str(e)) from e

        logger.info("Streaming synthesis completed", request_id=request_id)

    async def clone_voice(
        self,
        request: CloneVoiceRequest,
        audio_samples: list[bytes],
    ) -> VoiceInfo:
        """
        Clone a voice from audio samples.

        Args:
            request: Clone voice request with metadata
            audio_samples: List of audio sample bytes

        Returns:
            VoiceInfo for the newly created voice

        Raises:
            ValidationError: If samples are invalid
            AudioProcessingError: If cloning fails
        """
        with Timer("clone_voice", log=True, log_level="info"):
            if not audio_samples:
                raise ValidationError("At least one audio sample is required")

            if not request.voice_name:
                raise ValidationError("Voice name is required")

            logger.info(
                "Starting voice cloning",
                voice_name=request.voice_name,
                sample_count=len(audio_samples),
            )

            # Get engine
            engine = await self._get_engine()

            # Check if engine supports cloning
            capabilities = engine.get_capabilities()
            if not capabilities.supports_cloning:
                raise ValidationError("Voice cloning is not supported by this engine")

            # Build AudioSample objects
            samples: list[AudioSample] = []
            for i, audio_data in enumerate(audio_samples):
                sample_meta = (
                    request.samples[i] if i < len(request.samples) else None
                )
                samples.append(
                    AudioSample(
                        audio_data=audio_data,
                        format=sample_meta.format if sample_meta else "wav",
                        sample_rate=sample_meta.sample_rate if sample_meta else 22050,
                        duration_seconds=sample_meta.duration_seconds if sample_meta else 0,
                    )
                )

            # Clone voice
            try:
                voice = await engine.clone_voice(
                    samples=samples,
                    voice_name=request.voice_name,
                    description=request.description,
                )
            except NotImplementedError as e:
                raise ValidationError("Voice cloning is not implemented") from e
            except Exception as e:
                logger.error("Voice cloning failed", error=str(e))
                raise AudioProcessingError(
                    message="Voice cloning failed",
                    operation="clone",
                    details={"reason": str(e)},
                ) from e

            logger.info(
                "Voice cloned successfully",
                voice_id=voice.voice_id,
                voice_name=voice.name,
            )

        return voice

    async def list_voices(self, language: str | None = None) -> list[VoiceInfo]:
        """
        List available voices.

        Args:
            language: Optional language filter

        Returns:
            List of VoiceInfo
        """
        with Timer("list_voices", log=True, log_level="debug"):
            engine = await self._get_engine()
            voices = await engine.get_available_voices()

            # Filter by language if specified
            if language:
                voices = [v for v in voices if v.language == language]

            logger.debug(
                "Listed voices",
                total=len(voices),
                language_filter=language,
            )

        return voices

    async def get_voice(self, voice_id: str) -> VoiceInfo:
        """
        Get information about a specific voice.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceInfo

        Raises:
            VoiceNotFoundError: If voice doesn't exist
        """
        engine = await self._get_engine()
        voice = await engine.get_voice(voice_id)

        if voice is None:
            raise VoiceNotFoundError(voice_id)

        return voice

    async def delete_voice(self, voice_id: str) -> None:
        """
        Delete a cloned voice.

        Args:
            voice_id: Voice identifier to delete

        Raises:
            VoiceNotFoundError: If voice doesn't exist
            ValidationError: If voice cannot be deleted (built-in)
        """
        with Timer("delete_voice", log=True, log_level="info"):
            engine = await self._get_engine()

            # Check voice exists
            voice = await engine.get_voice(voice_id)
            if voice is None:
                raise VoiceNotFoundError(voice_id)

            # Check if it's a cloned voice
            if not voice.is_cloned:
                raise ValidationError("Cannot delete built-in voices")

            # Note: Actual deletion depends on engine implementation
            # For now, we log the intent - engine should handle storage
            logger.info("Voice deletion requested", voice_id=voice_id)

            # TODO: Call engine.delete_voice(voice_id) when implemented

    async def health_check(self) -> dict:
        """
        Check health of all engines.

        Returns:
            Dictionary with health information
        """
        with Timer("health_check", log=True, log_level="debug"):
            engines_health: dict[str, EngineHealthResponse] = {}
            overall_healthy = True

            # Check each loaded engine
            for engine_name in self._registry.list_loaded_engines():
                try:
                    engine = self._registry.get_engine_sync(engine_name)
                    if engine:
                        status = await engine.health_check()
                        engines_health[engine_name] = EngineHealthResponse(
                            name=engine_name,
                            healthy=status.healthy,
                            loaded=status.model_loaded,
                            latency_ms=status.latency_ms,
                            gpu_memory_used_mb=status.gpu_memory_used_mb,
                            gpu_memory_total_mb=status.gpu_memory_total_mb,
                            details=status.details,
                        )
                        if not status.healthy:
                            overall_healthy = False
                except Exception as e:
                    logger.warning(f"Health check failed for {engine_name}: {e}")
                    engines_health[engine_name] = EngineHealthResponse(
                        name=engine_name,
                        healthy=False,
                        loaded=False,
                        details={"error": str(e)},
                    )
                    overall_healthy = False

            # Determine overall status
            if not engines_health:
                status = "unhealthy"
            elif overall_healthy:
                status = "healthy"
            else:
                status = "degraded"

        return {
            "status": status,
            "uptime_seconds": self.uptime_seconds,
            "engines": {k: v.model_dump() for k, v in engines_health.items()},
        }

    def is_ready(self) -> bool:
        """Check if the service is ready to handle requests."""
        # Check if at least one engine is loaded
        loaded_engines = self._registry.list_loaded_engines()
        if not loaded_engines:
            return False

        # Check if the default engine is loaded and ready
        default_engine = self._registry.get_engine_sync()
        if default_engine is None:
            return False

        return default_engine.is_loaded


# Type alias for audio processor (to avoid circular import)
class AudioProcessor:
    """Placeholder for audio processor interface."""

    pass
