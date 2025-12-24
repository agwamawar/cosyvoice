"""Audio processing utilities combining validation, conversion, and metadata.

Provides:
- ProcessedAudio model for processed audio result
- AudioProcessor class combining validator and converter
"""

from pydantic import BaseModel, Field

from src.audio.converter import AudioConverter, ConversionOptions
from src.audio.formats import AudioFormat, detect_format_from_bytes
from src.audio.metadata import extract_metadata
from src.audio.validation import AudioRequirements, AudioValidator, ValidationResult
from src.utils.logging import get_logger

logger = get_logger("audio")


class ProcessedAudio(BaseModel):
    """Result of audio processing."""

    data: bytes = Field(..., description="Processed audio bytes")
    format: AudioFormat = Field(..., description="Audio format")
    sample_rate: int = Field(..., description="Sample rate in Hz")
    duration_seconds: float = Field(..., description="Duration in seconds")
    channels: int = Field(default=1, description="Number of channels")

    model_config = {"arbitrary_types_allowed": True}


class AudioProcessor:
    """
    High-level audio processor combining validation and conversion.

    Usage:
        processor = AudioProcessor()
        result = await processor.process_input(audio_bytes, requirements)
        output = await processor.process_output(audio_bytes, sample_rate, options)
    """

    def __init__(
        self,
        validator: AudioValidator | None = None,
        converter: AudioConverter | None = None,
    ) -> None:
        """
        Initialize the audio processor.

        Args:
            validator: AudioValidator instance (creates default if None)
            converter: AudioConverter instance (creates default if None)
        """
        self._validator = validator or AudioValidator()
        self._converter = converter or AudioConverter()

    @property
    def validator(self) -> AudioValidator:
        """Get the audio validator."""
        return self._validator

    @property
    def converter(self) -> AudioConverter:
        """Get the audio converter."""
        return self._converter

    async def process_input(
        self,
        audio_data: bytes,
        requirements: AudioRequirements | None = None,
    ) -> ProcessedAudio:
        """
        Process and validate input audio.

        Validates the audio against requirements, converts to standard format
        (mono WAV at target sample rate), and normalizes.

        Args:
            audio_data: Raw audio bytes
            requirements: Validation requirements

        Returns:
            ProcessedAudio with normalized audio

        Raises:
            ValueError: If audio fails validation
        """
        reqs = requirements or AudioRequirements()

        # Validate
        validation = self._validator.validate(audio_data, reqs)
        if not validation.valid:
            errors = "; ".join(validation.errors)
            raise ValueError(f"Audio validation failed: {errors}")

        # Detect format
        detected_format = validation.detected_format or AudioFormat.WAV

        # Convert to standard format
        options = ConversionOptions(
            target_format=AudioFormat.WAV,
            sample_rate=validation.detected_sample_rate or 22050,
            channels=1,  # Convert to mono
            normalize=True,
            normalize_db=-3.0,
        )

        processed_data = await self._converter.convert(
            audio_data,
            detected_format,
            options,
        )

        # Extract metadata from processed audio
        metadata = extract_metadata(processed_data)

        return ProcessedAudio(
            data=processed_data,
            format=AudioFormat.WAV,
            sample_rate=metadata.sample_rate,
            duration_seconds=metadata.duration_seconds,
            channels=metadata.channels,
        )

    async def process_output(
        self,
        audio_data: bytes,
        source_sample_rate: int,
        options: ConversionOptions,
    ) -> bytes:
        """
        Process output audio for delivery.

        Converts audio to target format with specified options.

        Args:
            audio_data: Raw audio bytes (assumed WAV)
            source_sample_rate: Source sample rate
            options: Conversion options

        Returns:
            Processed audio bytes in target format
        """
        # Detect source format
        source_format = detect_format_from_bytes(audio_data) or AudioFormat.WAV

        # Convert
        return await self._converter.convert(audio_data, source_format, options)

    async def validate(
        self,
        audio_data: bytes,
        requirements: AudioRequirements | None = None,
    ) -> ValidationResult:
        """
        Validate audio without processing.

        Args:
            audio_data: Raw audio bytes
            requirements: Validation requirements

        Returns:
            ValidationResult
        """
        return self._validator.validate(audio_data, requirements)

    async def convert(
        self,
        audio_data: bytes,
        options: ConversionOptions,
    ) -> bytes:
        """
        Convert audio format.

        Args:
            audio_data: Raw audio bytes
            options: Conversion options

        Returns:
            Converted audio bytes
        """
        source_format = detect_format_from_bytes(audio_data) or AudioFormat.WAV
        return await self._converter.convert(audio_data, source_format, options)

    async def resample(
        self,
        audio_data: bytes,
        source_rate: int,
        target_rate: int,
    ) -> bytes:
        """
        Resample audio.

        Args:
            audio_data: Raw audio bytes
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio bytes
        """
        return await self._converter.resample(audio_data, source_rate, target_rate)

    async def to_mono(self, audio_data: bytes) -> bytes:
        """
        Convert audio to mono.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Mono audio bytes
        """
        return await self._converter.to_mono(audio_data)

    async def normalize(
        self,
        audio_data: bytes,
        target_db: float = -3.0,
    ) -> bytes:
        """
        Normalize audio volume.

        Args:
            audio_data: Raw audio bytes
            target_db: Target peak dB level

        Returns:
            Normalized audio bytes
        """
        return await self._converter.normalize(audio_data, target_db)

    def get_duration(self, audio_data: bytes) -> float:
        """
        Get audio duration in seconds.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Duration in seconds
        """
        try:
            metadata = extract_metadata(audio_data)
            return metadata.duration_seconds
        except Exception:
            return 0.0

    def get_sample_rate(self, audio_data: bytes) -> int:
        """
        Get audio sample rate.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Sample rate in Hz
        """
        try:
            metadata = extract_metadata(audio_data)
            return metadata.sample_rate
        except Exception:
            return 22050


def create_audio_processor(
    default_sample_rate: int = 22050,
    default_requirements: AudioRequirements | None = None,
) -> AudioProcessor:
    """
    Create an AudioProcessor with default settings.

    Args:
        default_sample_rate: Default sample rate for conversion
        default_requirements: Default validation requirements

    Returns:
        Configured AudioProcessor
    """
    validator = AudioValidator(default_requirements)
    converter = AudioConverter(default_sample_rate)
    return AudioProcessor(validator, converter)
