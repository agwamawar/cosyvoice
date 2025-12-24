"""Audio format definitions and utilities.

Provides:
- AudioFormat enum for supported formats
- Format detection from bytes (magic bytes) and file extensions
- MIME type and extension utilities
- Default encoding settings per format
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class AudioFormat(str, Enum):
    """Supported audio formats."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"


@dataclass
class FormatInfo:
    """Information about an audio format."""

    name: str
    extension: str
    mime_type: str
    is_lossy: bool
    supports_streaming: bool
    typical_bitrate: int | None = None  # kbps for lossy formats
    magic_bytes: list[bytes] = field(default_factory=list)


# Magic bytes for format detection
# WAV: RIFF....WAVE
# MP3: ID3 or 0xFF 0xFB (frame sync)
# OGG: OggS
# FLAC: fLaC

FORMAT_INFO: dict[AudioFormat, FormatInfo] = {
    AudioFormat.WAV: FormatInfo(
        name="WAV",
        extension=".wav",
        mime_type="audio/wav",
        is_lossy=False,
        supports_streaming=True,
        magic_bytes=[b"RIFF", b"RIFX"],
    ),
    AudioFormat.MP3: FormatInfo(
        name="MP3",
        extension=".mp3",
        mime_type="audio/mpeg",
        is_lossy=True,
        supports_streaming=True,
        typical_bitrate=192,
        magic_bytes=[b"ID3", b"\xff\xfb", b"\xff\xfa", b"\xff\xf3", b"\xff\xf2"],
    ),
    AudioFormat.OGG: FormatInfo(
        name="OGG Vorbis",
        extension=".ogg",
        mime_type="audio/ogg",
        is_lossy=True,
        supports_streaming=True,
        typical_bitrate=128,
        magic_bytes=[b"OggS"],
    ),
    AudioFormat.FLAC: FormatInfo(
        name="FLAC",
        extension=".flac",
        mime_type="audio/flac",
        is_lossy=False,
        supports_streaming=False,
        magic_bytes=[b"fLaC"],
    ),
}


# Default encoding settings per format
DEFAULT_ENCODING_SETTINGS: dict[AudioFormat, dict[str, Any]] = {
    AudioFormat.WAV: {
        "encoding": "PCM_S",  # Signed PCM
        "bits_per_sample": 16,
    },
    AudioFormat.MP3: {
        "bitrate": 192000,  # 192 kbps
        "compression": -4.5,  # LAME quality (0=best, 9=worst)
    },
    AudioFormat.OGG: {
        "compression": 5,  # Vorbis quality (0-10, higher=better)
    },
    AudioFormat.FLAC: {
        "compression": 5,  # FLAC compression level (0-8, higher=smaller)
    },
}


def get_format_info(format: AudioFormat) -> FormatInfo:
    """Get information about an audio format."""
    return FORMAT_INFO[format]


def get_mime_type(format: AudioFormat) -> str:
    """Get MIME type for an audio format."""
    return FORMAT_INFO[format].mime_type


def get_file_extension(format: AudioFormat) -> str:
    """Get file extension for an audio format (with leading dot)."""
    return FORMAT_INFO[format].extension


def get_extension(format: AudioFormat) -> str:
    """Get file extension for an audio format (alias for get_file_extension)."""
    return FORMAT_INFO[format].extension


def get_encoding_settings(format: AudioFormat) -> dict[str, Any]:
    """Get default encoding settings for an audio format."""
    return DEFAULT_ENCODING_SETTINGS.get(format, {}).copy()


def detect_format_from_bytes(data: bytes) -> AudioFormat | None:
    """
    Detect audio format from magic bytes at the start of the data.

    Args:
        data: Audio file bytes (at least first 12 bytes needed)

    Returns:
        Detected AudioFormat or None if unrecognized
    """
    if len(data) < 4:
        return None

    # Check each format's magic bytes
    for fmt, info in FORMAT_INFO.items():
        for magic in info.magic_bytes:
            if data.startswith(magic):
                # Special check for WAV: must also contain WAVE
                if fmt == AudioFormat.WAV and len(data) >= 12:
                    if data[8:12] != b"WAVE":
                        continue
                return fmt

    # Additional check for MP3 without ID3 tag (frame sync)
    if len(data) >= 2:
        # Check for MP3 frame sync (11 bits set = 0xFF followed by 0xE0 or higher)
        if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
            return AudioFormat.MP3

    return None


def detect_format_from_extension(filename: str) -> AudioFormat | None:
    """
    Detect audio format from filename extension.

    Args:
        filename: Filename or path with extension

    Returns:
        Detected AudioFormat or None if unrecognized
    """
    path = Path(filename)
    ext = path.suffix.lower()

    if not ext:
        return None

    for fmt, info in FORMAT_INFO.items():
        if info.extension == ext:
            return fmt

    return None


def format_from_extension(extension: str) -> AudioFormat | None:
    """
    Get audio format from file extension.

    Args:
        extension: File extension (with or without leading dot)

    Returns:
        AudioFormat or None if unrecognized
    """
    ext = extension.lower()
    if not ext.startswith("."):
        ext = "." + ext

    for fmt, info in FORMAT_INFO.items():
        if info.extension == ext:
            return fmt

    return None


def format_from_mime_type(mime_type: str) -> AudioFormat | None:
    """
    Get audio format from MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        AudioFormat or None if unrecognized
    """
    mime = mime_type.lower().strip()

    # Handle common variations
    mime_aliases = {
        "audio/x-wav": "audio/wav",
        "audio/wave": "audio/wav",
        "audio/mp3": "audio/mpeg",
        "audio/x-mp3": "audio/mpeg",
        "audio/x-ogg": "audio/ogg",
        "audio/vorbis": "audio/ogg",
        "audio/x-flac": "audio/flac",
    }

    mime = mime_aliases.get(mime, mime)

    for fmt, info in FORMAT_INFO.items():
        if info.mime_type == mime:
            return fmt

    return None


def is_supported_format(format_str: str) -> bool:
    """Check if a format string is a supported audio format."""
    try:
        AudioFormat(format_str.lower())
        return True
    except ValueError:
        return False


def get_all_extensions() -> list[str]:
    """Get all supported file extensions."""
    return [info.extension for info in FORMAT_INFO.values()]


def get_all_mime_types() -> list[str]:
    """Get all supported MIME types."""
    return [info.mime_type for info in FORMAT_INFO.values()]
