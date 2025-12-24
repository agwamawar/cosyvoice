"""Audio format conversion utilities using torchaudio.

Provides:
- ConversionOptions model for conversion settings
- AudioConverter class with async methods for format conversion
- Resampling, mono conversion, and normalization
"""

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
import torchaudio
from pydantic import BaseModel, Field

from src.audio.formats import AudioFormat
from src.utils.logging import get_logger

logger = get_logger("audio")

# Thread pool for CPU-bound audio operations
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="audio_conv_")


class ConversionOptions(BaseModel):
    """Options for audio conversion."""

    target_format: AudioFormat = Field(default=AudioFormat.WAV)
    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=8)
    normalize: bool = Field(default=True)
    normalize_db: float = Field(default=-3.0, ge=-20.0, le=0.0)

    model_config = {"extra": "ignore"}


class AudioConverter:
    """
    Converts audio between formats using torchaudio.

    All operations run CPU-bound processing in a thread pool for async compatibility.
    """

    def __init__(self, default_sample_rate: int = 22050) -> None:
        self.default_sample_rate = default_sample_rate

    async def convert(
        self,
        audio_data: bytes,
        source_format: AudioFormat,
        options: ConversionOptions,
    ) -> bytes:
        """
        Convert audio to target format with specified options.

        Args:
            audio_data: Source audio bytes
            source_format: Source audio format
            options: Conversion options

        Returns:
            Converted audio bytes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            partial(self._convert_sync, audio_data, source_format, options),
        )

    def _convert_sync(
        self,
        audio_data: bytes,
        source_format: AudioFormat,
        options: ConversionOptions,
    ) -> bytes:
        """Synchronous conversion implementation."""
        # Load audio
        waveform, sample_rate = self._load_audio(audio_data)

        # Convert to mono if requested
        if options.channels == 1 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif options.channels == 2 and waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)

        # Resample if needed
        if sample_rate != options.sample_rate:
            waveform = self._resample_tensor(waveform, sample_rate, options.sample_rate)
            sample_rate = options.sample_rate

        # Normalize if requested
        if options.normalize:
            waveform = self._normalize_tensor(waveform, options.normalize_db)

        # Encode to target format
        return self._save_audio(waveform, sample_rate, options.target_format)

    async def resample(
        self,
        audio_data: bytes,
        source_rate: int,
        target_rate: int,
    ) -> bytes:
        """
        Resample audio to a different sample rate.

        Args:
            audio_data: Audio bytes
            source_rate: Source sample rate
            target_rate: Target sample rate

        Returns:
            Resampled audio bytes
        """
        if source_rate == target_rate:
            return audio_data

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            partial(self._resample_sync, audio_data, source_rate, target_rate),
        )

    def _resample_sync(
        self,
        audio_data: bytes,
        source_rate: int,
        target_rate: int,
    ) -> bytes:
        """Synchronous resample implementation."""
        waveform, _ = self._load_audio(audio_data)
        waveform = self._resample_tensor(waveform, source_rate, target_rate)
        return self._save_audio(waveform, target_rate, AudioFormat.WAV)

    async def to_mono(self, audio_data: bytes) -> bytes:
        """
        Convert audio to mono.

        Args:
            audio_data: Audio bytes

        Returns:
            Mono audio bytes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            partial(self._to_mono_sync, audio_data),
        )

    def _to_mono_sync(self, audio_data: bytes) -> bytes:
        """Synchronous mono conversion."""
        waveform, sample_rate = self._load_audio(audio_data)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return self._save_audio(waveform, sample_rate, AudioFormat.WAV)

    async def normalize(
        self,
        audio_data: bytes,
        target_db: float = -3.0,
    ) -> bytes:
        """
        Normalize audio volume.

        Args:
            audio_data: Audio bytes
            target_db: Target peak level in dB

        Returns:
            Normalized audio bytes
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            partial(self._normalize_sync, audio_data, target_db),
        )

    def _normalize_sync(self, audio_data: bytes, target_db: float) -> bytes:
        """Synchronous normalization."""
        waveform, sample_rate = self._load_audio(audio_data)
        waveform = self._normalize_tensor(waveform, target_db)
        return self._save_audio(waveform, sample_rate, AudioFormat.WAV)

    def _load_audio(self, audio_data: bytes) -> tuple[torch.Tensor, int]:
        """Load audio from bytes."""
        buffer = io.BytesIO(audio_data)
        waveform, sample_rate = torchaudio.load(buffer)
        return waveform, sample_rate

    def _save_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        target_format: AudioFormat,
    ) -> bytes:
        """Save audio to bytes in target format."""
        buffer = io.BytesIO()

        # Get format string for torchaudio
        format_map = {
            AudioFormat.WAV: "wav",
            AudioFormat.MP3: "mp3",
            AudioFormat.OGG: "ogg",
            AudioFormat.FLAC: "flac",
        }
        fmt = format_map.get(target_format, "wav")

        # torchaudio.save parameters
        kwargs = {}
        if target_format == AudioFormat.MP3:
            kwargs["compression"] = -4.5  # LAME quality
        elif target_format == AudioFormat.OGG:
            kwargs["compression"] = 5
        elif target_format == AudioFormat.FLAC:
            kwargs["compression"] = 5

        torchaudio.save(buffer, waveform, sample_rate, format=fmt, **kwargs)
        return buffer.getvalue()

    def _resample_tensor(
        self,
        waveform: torch.Tensor,
        source_rate: int,
        target_rate: int,
    ) -> torch.Tensor:
        """Resample a waveform tensor."""
        resampler = torchaudio.transforms.Resample(
            orig_freq=source_rate,
            new_freq=target_rate,
        )
        return resampler(waveform)

    def _normalize_tensor(
        self,
        waveform: torch.Tensor,
        target_db: float,
    ) -> torch.Tensor:
        """Normalize a waveform to target dB level."""
        # Find current peak
        peak = torch.abs(waveform).max()

        if peak == 0:
            return waveform

        # Calculate target linear amplitude from dB
        target_linear = 10 ** (target_db / 20)

        # Scale waveform
        scale = target_linear / peak
        return waveform * scale

    # Utility methods for compatibility
    @staticmethod
    def create_wav_header(
        num_samples: int,
        sample_rate: int,
        channels: int = 1,
        bits_per_sample: int = 16,
    ) -> bytes:
        """Create a WAV file header."""
        import struct

        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = num_samples * block_align

        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header

    @staticmethod
    def tensor_to_wav_bytes(
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> bytes:
        """Convert a waveform tensor to WAV bytes."""
        buffer = io.BytesIO()
        torchaudio.save(buffer, waveform, sample_rate, format="wav")
        return buffer.getvalue()

    @staticmethod
    def wav_bytes_to_tensor(wav_data: bytes) -> tuple[torch.Tensor, int]:
        """Convert WAV bytes to a waveform tensor."""
        buffer = io.BytesIO(wav_data)
        return torchaudio.load(buffer)
