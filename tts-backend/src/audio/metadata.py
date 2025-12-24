"""Audio metadata extraction using torchaudio.

Provides:
- AudioMetadata model for audio properties
- extract_metadata function using torchaudio
"""

import io
from pathlib import Path

import torchaudio
from pydantic import BaseModel, Field

from src.audio.formats import AudioFormat, detect_format_from_bytes, format_from_extension
from src.utils.logging import get_logger

logger = get_logger("audio")


class AudioMetadata(BaseModel):
    """Metadata for an audio file."""

    duration_seconds: float = Field(..., ge=0)
    sample_rate: int = Field(..., ge=1)
    channels: int = Field(..., ge=1)
    format: AudioFormat
    file_size_bytes: int = Field(..., ge=0)
    bits_per_sample: int = Field(default=16)
    bitrate_kbps: int | None = Field(default=None)

    model_config = {"extra": "ignore"}


def extract_metadata(audio_data: bytes) -> AudioMetadata:
    """
    Extract metadata from audio data using torchaudio.

    Args:
        audio_data: Raw audio bytes

    Returns:
        AudioMetadata object

    Raises:
        ValueError: If audio cannot be decoded
    """
    file_size = len(audio_data)

    if file_size == 0:
        raise ValueError("Audio data is empty")

    # Detect format
    detected_format = detect_format_from_bytes(audio_data)
    if detected_format is None:
        detected_format = AudioFormat.WAV

    # Load audio with torchaudio
    try:
        buffer = io.BytesIO(audio_data)
        waveform, sample_rate = torchaudio.load(buffer)

        channels = waveform.shape[0]
        num_samples = waveform.shape[1]
        duration_seconds = num_samples / sample_rate

        # Estimate bitrate for compressed formats
        bitrate_kbps: int | None = None
        if detected_format in (AudioFormat.MP3, AudioFormat.OGG):
            if duration_seconds > 0:
                bitrate_kbps = int((file_size * 8) / (duration_seconds * 1000))
        else:
            # For uncompressed, calculate from sample rate
            bits_per_sample = 16  # Assume 16-bit
            bitrate_kbps = (sample_rate * channels * bits_per_sample) // 1000

        return AudioMetadata(
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            format=detected_format,
            file_size_bytes=file_size,
            bits_per_sample=16,
            bitrate_kbps=bitrate_kbps,
        )

    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        raise ValueError(f"Failed to decode audio: {e}") from e


def extract_metadata_from_file(file_path: str | Path) -> AudioMetadata:
    """
    Extract metadata from an audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        AudioMetadata object
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    audio_data = path.read_bytes()
    return extract_metadata(audio_data)


def get_audio_info(audio_data: bytes) -> dict:
    """
    Get audio info as a dictionary.

    Args:
        audio_data: Raw audio bytes

    Returns:
        Dictionary with audio properties
    """
    try:
        metadata = extract_metadata(audio_data)
        return metadata.model_dump()
    except Exception as e:
        return {"error": str(e)}


def calculate_duration(
    num_samples: int,
    sample_rate: int,
) -> float:
    """Calculate audio duration from sample count."""
    if sample_rate <= 0:
        return 0.0
    return num_samples / sample_rate


def calculate_file_size(
    duration_seconds: float,
    sample_rate: int,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> int:
    """Estimate file size in bytes for uncompressed WAV audio."""
    bytes_per_sample = bits_per_sample // 8
    num_samples = int(duration_seconds * sample_rate)
    # WAV header (44 bytes) + audio data
    return 44 + num_samples * channels * bytes_per_sample


def samples_to_duration(num_samples: int, sample_rate: int) -> float:
    """Convert sample count to duration in seconds."""
    return num_samples / sample_rate if sample_rate > 0 else 0.0


def duration_to_samples(duration_seconds: float, sample_rate: int) -> int:
    """Convert duration to sample count."""
    return int(duration_seconds * sample_rate)
