"""Audio streaming utilities.

Provides:
- AudioChunker class for chunking audio data
- AudioBuffer class for buffering audio chunks
"""

import io
import struct
from collections.abc import Generator, Iterator

import torch
import torchaudio

from src.audio.formats import AudioFormat  # noqa: F401 - used in type hints
from src.utils.logging import get_logger

logger = get_logger("audio")


class AudioChunker:
    """
    Chunks audio data into smaller pieces for streaming.

    Usage:
        chunker = AudioChunker(sample_rate=22050)
        for chunk in chunker.chunk_audio(audio_bytes, chunk_duration_ms=100):
            yield chunk
    """

    def __init__(self, sample_rate: int = 22050, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels

    def chunk_audio(
        self,
        audio_data: bytes,
        chunk_duration_ms: int = 100,
    ) -> Iterator[bytes]:
        """
        Chunk audio data into smaller pieces.

        Args:
            audio_data: Raw audio bytes (WAV format)
            chunk_duration_ms: Duration of each chunk in milliseconds

        Yields:
            Audio chunks as bytes
        """
        # Load audio
        buffer = io.BytesIO(audio_data)
        try:
            waveform, sr = torchaudio.load(buffer)
        except Exception as e:
            logger.error(f"Failed to load audio for chunking: {e}")
            # Fall back to raw chunking
            yield from self._raw_chunk(audio_data, chunk_duration_ms)
            return

        # Calculate samples per chunk
        samples_per_chunk = int(sr * chunk_duration_ms / 1000)
        total_samples = waveform.shape[1]

        # Generate chunks
        for start in range(0, total_samples, samples_per_chunk):
            end = min(start + samples_per_chunk, total_samples)
            chunk_waveform = waveform[:, start:end]

            # Convert to bytes
            chunk_buffer = io.BytesIO()
            torchaudio.save(chunk_buffer, chunk_waveform, sr, format="wav")
            yield chunk_buffer.getvalue()

    def _raw_chunk(
        self,
        audio_data: bytes,
        chunk_duration_ms: int,
    ) -> Iterator[bytes]:
        """Raw byte-level chunking (fallback)."""
        # Assume 16-bit PCM
        bytes_per_sample = 2
        bytes_per_chunk = int(
            self.sample_rate * chunk_duration_ms / 1000 * self.channels * bytes_per_sample
        )

        # Skip WAV header if present
        start = 44 if audio_data[:4] == b"RIFF" else 0

        for i in range(start, len(audio_data), bytes_per_chunk):
            yield audio_data[i : i + bytes_per_chunk]

    def chunk_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        chunk_duration_ms: int = 100,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Chunk a waveform tensor.

        Args:
            waveform: Audio tensor (channels, samples)
            sample_rate: Sample rate
            chunk_duration_ms: Duration per chunk

        Yields:
            Chunk tensors
        """
        samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        total_samples = waveform.shape[1]

        for start in range(0, total_samples, samples_per_chunk):
            end = min(start + samples_per_chunk, total_samples)
            yield waveform[:, start:end]


class AudioBuffer:
    """
    Buffer for accumulating audio chunks.

    Usage:
        buffer = AudioBuffer(sample_rate=22050)
        buffer.add_chunk(chunk1)
        buffer.add_chunk(chunk2)
        full_audio = buffer.get_audio()
    """

    def __init__(self, sample_rate: int = 22050) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[bytes] = []
        self._total_bytes = 0
        self._waveform_chunks: list[torch.Tensor] = []

    def add_chunk(self, chunk: bytes) -> None:
        """
        Add a chunk to the buffer.

        Args:
            chunk: Audio chunk bytes
        """
        self._chunks.append(chunk)
        self._total_bytes += len(chunk)

    def add_waveform_chunk(self, chunk: torch.Tensor) -> None:
        """
        Add a waveform tensor chunk.

        Args:
            chunk: Waveform tensor (channels, samples)
        """
        self._waveform_chunks.append(chunk)

    def get_audio(self) -> bytes:
        """
        Get the complete audio as bytes.

        Returns:
            Complete audio bytes (WAV format)
        """
        if self._waveform_chunks:
            return self._get_audio_from_waveforms()
        return self._get_audio_from_bytes()

    def _get_audio_from_waveforms(self) -> bytes:
        """Concatenate waveform chunks and convert to bytes."""
        if not self._waveform_chunks:
            return b""

        # Concatenate all chunks
        full_waveform = torch.cat(self._waveform_chunks, dim=1)

        # Convert to WAV bytes
        buffer = io.BytesIO()
        torchaudio.save(buffer, full_waveform, self.sample_rate, format="wav")
        return buffer.getvalue()

    def _get_audio_from_bytes(self) -> bytes:
        """Concatenate byte chunks."""
        if not self._chunks:
            return b""

        # Try to load and concatenate properly
        waveforms = []
        for chunk in self._chunks:
            try:
                buffer = io.BytesIO(chunk)
                waveform, sr = torchaudio.load(buffer)
                waveforms.append(waveform)
            except Exception:
                # Skip invalid chunks
                continue

        if not waveforms:
            # Fall back to raw concatenation
            return b"".join(self._chunks)

        # Concatenate
        full_waveform = torch.cat(waveforms, dim=1)

        # Convert to WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, full_waveform, self.sample_rate, format="wav")
        return buffer.getvalue()

    def clear(self) -> None:
        """Clear the buffer."""
        self._chunks.clear()
        self._waveform_chunks.clear()
        self._total_bytes = 0

    @property
    def duration_seconds(self) -> float:
        """
        Estimated duration of buffered audio in seconds.

        Returns:
            Duration in seconds
        """
        if self._waveform_chunks:
            total_samples = sum(c.shape[1] for c in self._waveform_chunks)
            return total_samples / self.sample_rate

        # Estimate from bytes (assume 16-bit mono WAV)
        bytes_per_sample = 2
        estimated_samples = self._total_bytes // bytes_per_sample
        return estimated_samples / self.sample_rate

    @property
    def chunk_count(self) -> int:
        """Number of chunks in the buffer."""
        return len(self._chunks) + len(self._waveform_chunks)

    @property
    def total_bytes(self) -> int:
        """Total bytes in the buffer."""
        return self._total_bytes

    def __len__(self) -> int:
        """Return chunk count."""
        return self.chunk_count


def create_wav_header(
    num_samples: int,
    sample_rate: int,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Create a WAV file header.

    Args:
        num_samples: Number of samples
        sample_rate: Sample rate
        channels: Number of channels
        bits_per_sample: Bits per sample

    Returns:
        WAV header bytes (44 bytes)
    """
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
