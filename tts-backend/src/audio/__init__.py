"""Audio processing utilities using torchaudio.

Exports:
- Formats: AudioFormat, format utilities
- Validation: AudioValidator, AudioRequirements, ValidationResult
- Conversion: AudioConverter, ConversionOptions
- Metadata: AudioMetadata, extract_metadata
- Streaming: AudioChunker, AudioBuffer
- Processor: AudioProcessor, ProcessedAudio
"""

from src.audio.converter import AudioConverter, ConversionOptions
from src.audio.formats import (
    AudioFormat,
    FormatInfo,
    detect_format_from_bytes,
    detect_format_from_extension,
    format_from_extension,
    format_from_mime_type,
    get_encoding_settings,
    get_file_extension,
    get_format_info,
    get_mime_type,
)
from src.audio.metadata import (
    AudioMetadata,
    calculate_duration,
    calculate_file_size,
    extract_metadata,
    extract_metadata_from_file,
)
from src.audio.processor import AudioProcessor, ProcessedAudio, create_audio_processor
from src.audio.streaming import AudioBuffer, AudioChunker, create_wav_header
from src.audio.validation import (
    AudioRequirements,
    AudioValidator,
    ValidationResult,
    validate_audio,
)

__all__ = [
    # Formats
    "AudioFormat",
    "FormatInfo",
    "get_format_info",
    "get_mime_type",
    "get_file_extension",
    "get_encoding_settings",
    "detect_format_from_bytes",
    "detect_format_from_extension",
    "format_from_extension",
    "format_from_mime_type",
    # Validation
    "AudioValidator",
    "AudioRequirements",
    "ValidationResult",
    "validate_audio",
    # Conversion
    "AudioConverter",
    "ConversionOptions",
    # Metadata
    "AudioMetadata",
    "extract_metadata",
    "extract_metadata_from_file",
    "calculate_duration",
    "calculate_file_size",
    # Streaming
    "AudioChunker",
    "AudioBuffer",
    "create_wav_header",
    # Processor
    "AudioProcessor",
    "ProcessedAudio",
    "create_audio_processor",
]
