"""Test factories for generating test data.

Provides:
- SynthesisRequestFactory: generates valid synthesis requests
- VoiceInfoFactory: generates voice info objects
- create_audio_sample(): creates actual audio bytes
"""

import io
import math
import struct
import wave
from typing import Any, Literal

from src.core.interfaces import AudioResult, VoiceInfo


# -----------------------------------------------------------------------------
# Synthesis Request Factory
# -----------------------------------------------------------------------------
class SynthesisRequestFactory:
    """Factory for generating synthesis request data."""

    _counter = 0

    @classmethod
    def create(
        cls,
        text: str | None = None,
        voice_id: str = "default",
        output_format: str = "wav",
        sample_rate: int = 22050,
        speed: float = 1.0,
        language: str = "en",
    ) -> dict[str, Any]:
        """Create a synthesis request dictionary."""
        cls._counter += 1
        return {
            "text": text or f"Test synthesis request number {cls._counter}.",
            "voice_id": voice_id,
            "output_format": output_format,
            "sample_rate": sample_rate,
            "speed": speed,
            "language": language,
        }

    @classmethod
    def create_batch(cls, count: int = 3) -> list[dict[str, Any]]:
        """Create multiple synthesis requests."""
        return [cls.create() for _ in range(count)]


# -----------------------------------------------------------------------------
# Voice Info Factory
# -----------------------------------------------------------------------------
class VoiceInfoFactory:
    """Factory for generating VoiceInfo objects."""

    _counter = 0

    @classmethod
    def create(
        cls,
        voice_id: str | None = None,
        name: str | None = None,
        language: str = "en",
        gender: Literal["male", "female", "neutral"] | None = "neutral",
        is_cloned: bool = False,
        description: str | None = None,
    ) -> VoiceInfo:
        """Create a VoiceInfo instance."""
        cls._counter += 1
        return VoiceInfo(
            voice_id=voice_id or f"voice-{cls._counter}",
            name=name or f"Test Voice {cls._counter}",
            language=language,
            gender=gender,
            is_cloned=is_cloned,
            description=description or f"Test voice description {cls._counter}",
        )

    @classmethod
    def create_default_voices(cls) -> list[VoiceInfo]:
        """Create list of default voices matching the engine defaults."""
        return [
            cls.create(voice_id="default", name="Default", language="en", gender="neutral"),
            cls.create(voice_id="en-female-1", name="English Female 1", language="en", gender="female"),
            cls.create(voice_id="en-male-1", name="English Male 1", language="en", gender="male"),
            cls.create(voice_id="zh-female-1", name="Chinese Female 1", language="zh", gender="female"),
            cls.create(voice_id="zh-male-1", name="Chinese Male 1", language="zh", gender="male"),
        ]


# -----------------------------------------------------------------------------
# Audio Sample Creation
# -----------------------------------------------------------------------------
def create_audio_sample(
    duration: float = 3.0,
    sample_rate: int = 22050,
    format: Literal["wav", "raw"] = "wav",
    frequency: float = 440.0,
    silence: bool = False,
) -> bytes:
    """
    Create actual audio bytes for testing.

    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        format: Output format ("wav" or "raw")
        frequency: Tone frequency in Hz (ignored if silence=True)
        silence: If True, generate silence instead of tone

    Returns:
        Audio data as bytes
    """
    num_samples = int(sample_rate * duration)

    # Generate samples
    if silence:
        samples = [0] * num_samples
    else:
        # Generate sine wave
        samples = [
            int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / sample_rate))
            for i in range(num_samples)
        ]

    if format == "raw":
        return struct.pack(f"<{num_samples}h", *samples)

    # Create WAV file
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{num_samples}h", *samples))

    return buffer.getvalue()


def create_invalid_audio() -> bytes:
    """Create invalid/corrupted audio data."""
    return b"RIFF\x00\x00\x00\x00WAVEfmt corrupted data here"


def create_empty_wav() -> bytes:
    """Create an empty WAV file (header only, no samples)."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"")
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def create_synthesis_request(**kwargs: Any) -> dict[str, Any]:
    """Convenience function to create synthesis request."""
    return SynthesisRequestFactory.create(**kwargs)


def create_voice_info(**kwargs: Any) -> VoiceInfo:
    """Convenience function to create voice info."""
    return VoiceInfoFactory.create(**kwargs)


def create_voice(**kwargs: Any) -> VoiceInfo:
    """Alias for create_voice_info."""
    return VoiceInfoFactory.create(**kwargs)


def create_audio_result(
    duration_seconds: float = 1.0,
    sample_rate: int = 22050,
    format: str = "wav",
    channels: int = 1,
) -> AudioResult:
    """Create an AudioResult for testing."""
    audio_data = create_audio_sample(duration=duration_seconds, sample_rate=sample_rate)
    return AudioResult(
        audio_data=audio_data,
        format=format,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        channels=channels,
    )
