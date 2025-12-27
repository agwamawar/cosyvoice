"""CosyVoice TTS engine implementation.

Provides:
- CosyVoiceEngine implementing TTSEngine interface
- Uses real or mock inference based on config
- Fully implements all interface methods
"""

import io
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path

import torch
import torchaudio

from src.audio.formats import AudioFormat
from src.core.interfaces import (
    AudioResult,
    AudioSample,
    EngineCapabilities,
    HealthStatus,
    SynthesisOptions,
    TTSEngine,
    VoiceInfo,
)
from src.models.cosyvoice.config import CosyVoiceConfig
from src.models.cosyvoice.inference import (
    CosyVoiceInference,
    MockCosyVoiceInference,
    get_inference,
)
from src.models.cosyvoice.preprocessor import TextPreprocessor, get_preprocessor
from src.models.cosyvoice.voice_manager import VoiceManager
from src.utils.logging import get_logger

logger = get_logger("inference")


class CosyVoiceEngine(TTSEngine):
    """
    CosyVoice 3 TTS Engine implementing TTSEngine interface.

    Uses mock inference for local testing, real inference on RunPod.
    Mock mode works fully; real mode raises NotImplementedError until RunPod.
    """

    def __init__(self, config: CosyVoiceConfig | None = None) -> None:
        """
        Initialize CosyVoice engine.

        Args:
            config: Engine configuration (uses defaults if None)
        """
        self._config = config or CosyVoiceConfig()
        self._inference: CosyVoiceInference | MockCosyVoiceInference | None = None
        self._voice_manager: VoiceManager | None = None
        self._preprocessor: TextPreprocessor | None = None
        self._loaded = False

    @property
    def name(self) -> str:
        """Engine name identifier."""
        return "cosyvoice"

    @property
    def is_loaded(self) -> bool:
        """Check if engine is loaded."""
        return self._loaded

    async def load(self) -> None:
        """Load the engine and all components."""
        logger.info(f"Loading CosyVoice engine (mock={self._config.use_mock})...")

        # Initialize preprocessor
        self._preprocessor = get_preprocessor("en")

        # Initialize voice manager
        self._voice_manager = VoiceManager(
            voices_directory="./voices",
            embedding_dim=self._config.speaker_embedding_dim,
            device=self._config.get_device(),
        )
        await self._voice_manager.load_voices()

        # Initialize inference (mock or real based on config)
        self._inference = get_inference(self._config)
        await self._inference.load_model()

        self._loaded = True
        logger.info("CosyVoice engine loaded successfully")

    async def unload(self) -> None:
        """Unload the engine and free resources."""
        logger.info("Unloading CosyVoice engine...")

        if self._inference:
            await self._inference.unload_model()
            self._inference = None

        self._voice_manager = None
        self._preprocessor = None
        self._loaded = False

        logger.info("CosyVoice engine unloaded")

    async def synthesize(
        self,
        text: str,
        options: SynthesisOptions,
    ) -> AudioResult:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            options: Synthesis options

        Returns:
            AudioResult with synthesized audio
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        # Preprocess text
        language = options.language or "en"
        processed_text = self._preprocessor.preprocess(text, language)

        # Split long text if needed (no language tags - using instruct2 for language control)
        chunks = self._preprocessor.split_long_text(
            processed_text,
            self._config.max_text_length,
        )

        # Get speaker embedding (for compatibility) and prompt audio path
        embedding = await self._voice_manager.get_embedding(options.voice_id)
        prompt_audio_path = await self._voice_manager.get_reference_audio_path(options.voice_id)

        # Convert Path to string if exists
        prompt_audio_str = str(prompt_audio_path) if prompt_audio_path else None

        logger.info(
            f"Synthesizing: voice={options.voice_id}, language={language}, "
            f"prompt_audio={prompt_audio_str}, chunks={len(chunks)}"
        )

        # Generate audio for each chunk
        audio_chunks = []
        for chunk in chunks:
            audio_tensor, sample_rate = await self._inference.generate(
                chunk,
                embedding,
                speed=options.speed,
                prompt_audio_path=prompt_audio_str,
                language=language,
            )
            audio_chunks.append(audio_tensor)

        # Concatenate chunks
        if len(audio_chunks) > 1:
            full_audio = torch.cat(audio_chunks, dim=1)
        else:
            full_audio = audio_chunks[0]

        # Convert to target format
        audio_bytes = self._tensor_to_bytes(
            full_audio,
            sample_rate,
            options.output_format,
        )

        # Calculate duration
        duration = full_audio.shape[1] / sample_rate

        return AudioResult(
            audio_data=audio_bytes,
            format=options.output_format,
            sample_rate=options.sample_rate or sample_rate,
            duration_seconds=duration,
            metadata={
                "voice_id": options.voice_id,
                "language": language,
                "text_length": len(text),
            },
        )

    async def synthesize_stream(
        self,
        text: str,
        options: SynthesisOptions,
        chunk_size_ms: int = 100,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesized speech.

        Args:
            text: Text to synthesize
            options: Synthesis options
            chunk_size_ms: Chunk duration in milliseconds

        Yields:
            Audio chunks as bytes
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        # For now, generate full audio and chunk it
        result = await self.synthesize(text, options)

        # Load audio for chunking
        buffer = io.BytesIO(result.audio_data)
        waveform, sr = torchaudio.load(buffer)

        # Calculate samples per chunk
        samples_per_chunk = int(sr * chunk_size_ms / 1000)
        total_samples = waveform.shape[1]

        # Yield WAV header first
        yield self._create_wav_header(total_samples, sr)

        # Yield chunks
        for start in range(0, total_samples, samples_per_chunk):
            end = min(start + samples_per_chunk, total_samples)
            chunk = waveform[:, start:end]

            # Convert to bytes (raw PCM)
            chunk_bytes = (chunk * 32767).to(torch.int16).numpy().tobytes()
            yield chunk_bytes

    async def clone_voice(
        self,
        samples: list[AudioSample],
        voice_name: str,
        description: str | None = None,
    ) -> VoiceInfo:
        """
        Clone a voice from audio samples.

        Args:
            samples: Audio samples for cloning
            voice_name: Name for the new voice
            description: Optional description

        Returns:
            VoiceInfo for the cloned voice
        """
        if not self._loaded:
            raise RuntimeError("Engine not loaded")

        # Use first sample for embedding extraction
        if not samples:
            raise ValueError("At least one audio sample is required")

        sample = samples[0]

        # Save voice first to get voice_id and directory
        metadata = {
            "description": description,
            "language": "en",
            "sample_duration": sample.duration_seconds,
        }
        # Save with None embedding initially - we'll use reference audio directly
        voice_id = await self._voice_manager.save_voice(voice_name, None, metadata)

        # Save reference audio to the voice directory
        reference_path = await self._voice_manager.save_reference_audio(
            voice_id, sample.audio_data
        )

        logger.info(f"Saved reference audio for voice {voice_id} at {reference_path}")

        # Register speaker with CosyVoice using the saved audio path
        try:
            speaker_id = await self._inference.extract_speaker_embedding(
                str(reference_path),
                speaker_name=voice_id,
            )
            logger.info(f"Registered speaker: {speaker_id}")
        except Exception as e:
            logger.warning(f"Speaker registration failed (non-fatal): {e}")
            # Voice cloning will still work using cross-lingual with reference audio

        # Get voice info
        voice = await self._voice_manager.get_voice(voice_id)
        return voice

    async def get_available_voices(self) -> list[VoiceInfo]:
        """Get list of available voices."""
        if not self._voice_manager:
            return []
        return await self._voice_manager.list_voices()

    def get_capabilities(self) -> EngineCapabilities:
        """Get engine capabilities."""
        return EngineCapabilities(
            supports_cloning=True,
            supports_streaming=True,
            supports_emotions=False,
            supported_languages=self._config.supported_languages,
            max_text_length=self._config.max_text_length,
        )

    async def health_check(self) -> HealthStatus:
        """Check engine health."""
        gpu_memory_used = None
        gpu_memory_total = None

        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

        return HealthStatus(
            healthy=self._loaded,
            latency_ms=0.0,  # Would measure actual inference latency
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            model_loaded=self._loaded,
            details={
                "engine": "cosyvoice",
                "mock_mode": self._config.use_mock,
                "device": self._config.get_device(),
            },
        )

    def _tensor_to_bytes(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        output_format: AudioFormat,
    ) -> bytes:
        """Convert audio tensor to bytes in target format."""
        buffer = io.BytesIO()

        format_map = {
            AudioFormat.WAV: "wav",
            AudioFormat.MP3: "mp3",
            AudioFormat.OGG: "ogg",
            AudioFormat.FLAC: "flac",
        }
        fmt = format_map.get(output_format, "wav")

        torchaudio.save(buffer, audio, sample_rate, format=fmt)
        return buffer.getvalue()

    def _create_wav_header(self, num_samples: int, sample_rate: int) -> bytes:
        """Create WAV header for streaming."""
        import struct

        data_size = num_samples * 2  # 16-bit mono
        return struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,  # PCM
            1,  # mono
            sample_rate,
            sample_rate * 2,
            2,
            16,
            b"data",
            data_size,
        )
