"""F5-TTS inference utilities.

Provides:
- F5TTSInference: Real F5-TTS inference using the f5_tts package
- MockF5TTSInference: Mock inference for local testing
"""

import asyncio
import math
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from src.models.f5tts.config import F5TTSConfig
from src.utils.logging import get_logger

logger = get_logger("inference")

# Thread pool for CPU/GPU-bound inference operations
_inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="f5tts_inference")


class F5TTSInference:
    """
    Real F5-TTS inference handler.

    Uses the official F5-TTS package for inference.
    Supports zero-shot voice cloning with reference audio.
    """

    def __init__(self, config: F5TTSConfig) -> None:
        """
        Initialize inference handler.

        Args:
            config: F5-TTS configuration
        """
        self.config = config
        self._model: Any = None
        self._loaded: bool = False
        self._sample_rate: int = 24000  # F5-TTS default

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def sample_rate(self) -> int:
        """Get the model's output sample rate."""
        return self._sample_rate

    async def load_model(self) -> None:
        """
        Load the F5-TTS model.

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Loading F5-TTS model...")

        # Run loading in thread pool (blocking operation)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_inference_executor, self._load_model_sync)

        self._loaded = True
        logger.info(f"F5-TTS model loaded successfully (sample_rate={self._sample_rate})")

    def _load_model_sync(self) -> None:
        """Synchronous model loading (runs in thread pool)."""
        try:
            from f5_tts.api import F5TTS

            logger.info("Initializing F5-TTS...")

            # Initialize F5-TTS
            # The model will be downloaded automatically if not present
            self._model = F5TTS(
                model_type=self.config.model_name,
                device=self.config.get_device(),
            )

            # F5-TTS uses 24kHz sample rate
            self._sample_rate = 24000

            logger.info(f"Using device: {self.config.get_device()}")

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import F5-TTS. Install with: pip install f5-tts. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load F5-TTS model: {e}") from e

    async def unload_model(self) -> None:
        """Unload model and clear CUDA cache."""
        if self._model is not None:
            del self._model
            self._model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("F5-TTS model unloaded")

    async def generate(
        self,
        text: str,
        speaker_embedding: Any,
        speed: float = 1.0,
        prompt_audio_path: str | None = None,
        prompt_text: str = "",
        language: str = "en",
    ) -> tuple[torch.Tensor, int]:
        """
        Generate audio from text using F5-TTS voice cloning.

        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding (unused, kept for compatibility)
            speed: Speech speed multiplier (0.5-2.0)
            prompt_audio_path: Path to reference audio file for voice cloning
            prompt_text: Transcript of the reference audio
            language: Language code (unused - F5-TTS auto-detects)

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        if not prompt_audio_path or not Path(prompt_audio_path).exists():
            raise RuntimeError(
                "F5-TTS requires reference audio for synthesis. "
                "Please provide a voice with reference audio or clone a voice first."
            )

        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            _inference_executor,
            self._generate_sync,
            text,
            prompt_audio_path,
            prompt_text,
            speed,
        )

        return audio_tensor, self._sample_rate

    def _generate_sync(
        self,
        text: str,
        prompt_audio_path: str,
        prompt_text: str,
        speed: float,
    ) -> torch.Tensor:
        """
        Synchronous generation using F5-TTS.

        Args:
            text: Text to synthesize
            prompt_audio_path: Path to reference audio
            prompt_text: Transcript of reference audio
            speed: Speech speed multiplier
        """
        logger.info(f"Using F5-TTS inference: prompt={prompt_audio_path}")
        logger.info(f"Text (first 100 chars): {text[:100]}...")

        # F5-TTS infer method
        # Returns: (audio_array, sample_rate, spectrogram)
        audio, sr, _ = self._model.infer(
            ref_file=prompt_audio_path,
            ref_text=prompt_text,
            gen_text=text,
            nfe_step=self.config.nfe_step,
            cfg_strength=self.config.cfg_strength,
            sway_sampling_coef=self.config.sway_sampling_coef,
            speed=speed,
            cross_fade_duration=self.config.cross_fade_duration,
        )

        # Convert to torch tensor if needed
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)

        # Ensure correct shape (1, samples) for mono audio
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return audio

    async def generate_with_file_output(
        self,
        text: str,
        prompt_audio_path: str,
        prompt_text: str,
        output_path: str,
        speed: float = 1.0,
    ) -> str:
        """
        Generate audio and save to file.

        Args:
            text: Text to synthesize
            prompt_audio_path: Path to reference audio
            prompt_text: Transcript of reference audio
            output_path: Path to save output audio
            speed: Speech speed multiplier

        Returns:
            Path to output audio file
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _inference_executor,
            self._generate_to_file_sync,
            text,
            prompt_audio_path,
            prompt_text,
            output_path,
            speed,
        )

        return output_path

    def _generate_to_file_sync(
        self,
        text: str,
        prompt_audio_path: str,
        prompt_text: str,
        output_path: str,
        speed: float,
    ) -> None:
        """Synchronous generation with file output."""
        self._model.infer(
            ref_file=prompt_audio_path,
            ref_text=prompt_text,
            gen_text=text,
            file_wave=output_path,
            nfe_step=self.config.nfe_step,
            cfg_strength=self.config.cfg_strength,
            sway_sampling_coef=self.config.sway_sampling_coef,
            speed=speed,
            cross_fade_duration=self.config.cross_fade_duration,
        )

    async def extract_speaker_embedding(
        self,
        audio_path: str,
        speaker_name: str | None = None,
    ) -> str:
        """
        Register a speaker from audio (F5-TTS doesn't need explicit registration).

        F5-TTS uses the reference audio directly during inference,
        so we just return a speaker ID for compatibility.

        Args:
            audio_path: Path to reference audio file
            speaker_name: Optional name for the speaker

        Returns:
            Speaker ID string
        """
        if speaker_name is None:
            speaker_name = f"speaker_{uuid.uuid4().hex[:8]}"

        logger.info(f"Registered speaker: {speaker_name} (F5-TTS uses audio directly)")
        return speaker_name


class MockF5TTSInference:
    """
    Mock F5-TTS inference for testing without GPU.

    Generates sine wave audio for testing the API pipeline.
    """

    def __init__(self, config: F5TTSConfig) -> None:
        """
        Initialize mock inference.

        Args:
            config: F5-TTS configuration
        """
        self.config = config
        self._loaded: bool = False
        self._sample_rate: int = 24000

    @property
    def is_loaded(self) -> bool:
        """Check if mock is loaded."""
        return self._loaded

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    async def load_model(self) -> None:
        """Mock model loading (instant)."""
        logger.info("Loading mock F5-TTS inference...")
        self._loaded = True
        logger.info("Mock F5-TTS inference loaded")

    async def unload_model(self) -> None:
        """Mock model unloading."""
        self._loaded = False
        logger.info("Mock F5-TTS inference unloaded")

    def _generate_mock_audio(self, text: str, speed: float = 1.0) -> torch.Tensor:
        """Generate mock sine wave audio."""
        sample_rate = self._sample_rate

        # Estimate duration: ~100ms per word, adjusted by speed
        word_count = len(text.split())
        duration_seconds = max(0.5, (word_count * 0.1) / speed)

        # Generate sine wave
        num_samples = int(sample_rate * duration_seconds)
        t = torch.linspace(0, duration_seconds, num_samples)

        # Use voice-like frequency with harmonics
        frequency = 150.0  # Fundamental frequency
        audio = torch.zeros(num_samples)

        # Add harmonics for more natural sound
        for harmonic in [1, 2, 3, 4]:
            amplitude = 0.3 / harmonic
            audio += amplitude * torch.sin(2 * math.pi * frequency * harmonic * t)

        # Apply envelope (fade in/out)
        fade_samples = int(sample_rate * 0.02)  # 20ms fade
        fade_in = torch.linspace(0, 1, fade_samples)
        fade_out = torch.linspace(1, 0, fade_samples)

        if len(audio) > 2 * fade_samples:
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out

        # Add shape for mono audio (1, samples)
        return audio.unsqueeze(0)

    async def generate(
        self,
        text: str,
        speaker_embedding: Any,
        speed: float = 1.0,
        prompt_audio_path: str | None = None,
        prompt_text: str = "",
        language: str = "en",
    ) -> tuple[torch.Tensor, int]:
        """
        Generate mock audio (sine wave).

        Args:
            text: Text to synthesize
            speaker_embedding: Ignored for mock
            speed: Speech speed multiplier
            prompt_audio_path: Ignored for mock
            prompt_text: Ignored for mock
            language: Ignored for mock

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        audio = self._generate_mock_audio(text, speed)
        return audio, self._sample_rate

    async def extract_speaker_embedding(
        self,
        audio_path: str,
        speaker_name: str | None = None,
    ) -> str:
        """Mock speaker embedding extraction."""
        if speaker_name is None:
            speaker_name = f"mock_speaker_{uuid.uuid4().hex[:8]}"
        return speaker_name


def get_inference(config: F5TTSConfig) -> F5TTSInference | MockF5TTSInference:
    """
    Get appropriate inference handler based on config.

    Args:
        config: F5-TTS configuration

    Returns:
        F5TTSInference or MockF5TTSInference
    """
    if config.use_mock:
        return MockF5TTSInference(config)
    return F5TTSInference(config)
