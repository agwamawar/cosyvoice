"""Audio validation utilities using torchaudio.

Provides:
- AudioRequirements model for validation configuration
- ValidationResult model with detailed results
- AudioValidator class using torchaudio for audio inspection
"""

import io

import torch
import torchaudio
from pydantic import BaseModel, Field

from src.audio.formats import AudioFormat, detect_format_from_bytes
from src.utils.logging import get_logger

logger = get_logger("audio")


class AudioRequirements(BaseModel):
    """Requirements for audio validation."""

    min_duration_seconds: float = Field(default=1.0, ge=0)
    max_duration_seconds: float = Field(default=300.0, ge=0)
    min_sample_rate: int = Field(default=8000, ge=1000)
    max_sample_rate: int = Field(default=48000, le=192000)
    allowed_formats: list[AudioFormat] = Field(
        default_factory=lambda: list(AudioFormat)
    )
    max_file_size_bytes: int = Field(default=50 * 1024 * 1024)  # 50MB
    require_mono: bool = Field(default=False)
    max_channels: int = Field(default=8)

    model_config = {"extra": "ignore"}


class ValidationResult(BaseModel):
    """Result of audio validation."""

    valid: bool = Field(..., description="Whether the audio passed all checks")
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    detected_format: AudioFormat | None = Field(default=None)
    detected_duration: float | None = Field(default=None)
    detected_sample_rate: int | None = Field(default=None)
    detected_channels: int | None = Field(default=None)
    file_size_bytes: int | None = Field(default=None)

    model_config = {"extra": "ignore"}


class AudioValidator:
    """
    Validates audio data against requirements using torchaudio.

    Handles errors gracefully - corrupted files return valid=False
    with an error message rather than raising exceptions.
    """

    def __init__(self, requirements: AudioRequirements | None = None) -> None:
        self._default_requirements = requirements or AudioRequirements()

    def validate(
        self,
        audio_data: bytes,
        requirements: AudioRequirements | None = None,
    ) -> ValidationResult:
        """Validate audio data against requirements."""
        reqs = requirements or self._default_requirements
        errors: list[str] = []
        warnings: list[str] = []

        file_size = len(audio_data)

        if file_size == 0:
            return ValidationResult(
                valid=False,
                errors=["Audio data is empty"],
                file_size_bytes=0,
            )

        # Check file size
        if file_size > reqs.max_file_size_bytes:
            size_mb = file_size / (1024 * 1024)
            max_mb = reqs.max_file_size_bytes / (1024 * 1024)
            errors.append(f"File size ({size_mb:.1f}MB) exceeds maximum ({max_mb:.1f}MB)")

        # Detect format
        detected_format = detect_format_from_bytes(audio_data)
        if detected_format is None:
            errors.append("Could not detect audio format from file header")
        elif detected_format not in reqs.allowed_formats:
            allowed = ", ".join(f.value for f in reqs.allowed_formats)
            errors.append(f"Format '{detected_format.value}' not allowed. Allowed: {allowed}")

        # Load and inspect audio with torchaudio
        duration: float | None = None
        sample_rate: int | None = None
        channels: int | None = None

        try:
            waveform, sr = self._load_audio(audio_data)
            sample_rate = sr
            channels = waveform.shape[0]
            num_samples = waveform.shape[1]
            duration = num_samples / sample_rate

            # Validate sample rate
            if sample_rate < reqs.min_sample_rate:
                errors.append(f"Sample rate ({sample_rate}Hz) below minimum ({reqs.min_sample_rate}Hz)")
            elif sample_rate > reqs.max_sample_rate:
                errors.append(f"Sample rate ({sample_rate}Hz) exceeds maximum ({reqs.max_sample_rate}Hz)")

            # Validate duration
            if duration < reqs.min_duration_seconds:
                errors.append(f"Duration ({duration:.2f}s) below minimum ({reqs.min_duration_seconds}s)")
            elif duration > reqs.max_duration_seconds:
                errors.append(f"Duration ({duration:.2f}s) exceeds maximum ({reqs.max_duration_seconds}s)")

            # Validate channels
            if channels > reqs.max_channels:
                errors.append(f"Audio has {channels} channels, maximum is {reqs.max_channels}")

            if reqs.require_mono and channels > 1:
                warnings.append(f"Audio has {channels} channels but mono required. Will convert.")

            if channels > 2:
                warnings.append(f"Audio has {channels} channels (surround). Consider stereo/mono.")

            # Check for silence
            max_amplitude = torch.abs(waveform).max().item()
            if max_amplitude < 0.001:
                warnings.append("Audio appears silent or nearly silent")
            elif max_amplitude < 0.01:
                warnings.append("Audio has very low volume")

            if max_amplitude > 0.99:
                warnings.append("Audio may be clipped (max amplitude near 1.0)")

        except Exception as e:
            logger.warning(f"Failed to load audio for validation: {e}")
            errors.append(f"Failed to decode audio: {str(e)}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            detected_format=detected_format,
            detected_duration=duration,
            detected_sample_rate=sample_rate,
            detected_channels=channels,
            file_size_bytes=file_size,
        )

    def _load_audio(self, audio_data: bytes) -> tuple[torch.Tensor, int]:
        """Load audio data using torchaudio."""
        buffer = io.BytesIO(audio_data)
        waveform, sample_rate = torchaudio.load(buffer)
        return waveform, sample_rate

    def validate_for_cloning(
        self,
        audio_data: bytes,
        min_duration: float = 3.0,
        max_duration: float = 30.0,
    ) -> ValidationResult:
        """Validate audio specifically for voice cloning."""
        requirements = AudioRequirements(
            min_duration_seconds=min_duration,
            max_duration_seconds=max_duration,
            min_sample_rate=16000,
            max_sample_rate=48000,
            max_file_size_bytes=20 * 1024 * 1024,
        )
        return self.validate(audio_data, requirements)


def validate_audio(
    audio_data: bytes,
    requirements: AudioRequirements | None = None,
) -> ValidationResult:
    """Convenience function to validate audio data."""
    validator = AudioValidator()
    return validator.validate(audio_data, requirements)


# Alias for backward compatibility
ValidationConfig = AudioRequirements
