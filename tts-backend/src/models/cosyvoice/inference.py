"""CosyVoice inference utilities.

Provides:
- CosyVoiceInference: Real CosyVoice 3 inference
- MockCosyVoiceInference: Mock inference for local testing
"""

import asyncio
import math
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from src.models.cosyvoice.config import CosyVoiceConfig
from src.utils.logging import get_logger

logger = get_logger("inference")

# Thread pool for CPU/GPU-bound inference operations
_inference_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cosyvoice_inference")

# NOTE: CosyVoice3 does NOT use language tags like <|en|> or <|zh|>
# Those are for OLD CosyVoice (CosyVoice-300M) only!
# CosyVoice3 auto-detects language from the input text itself.
# Just pass plain text - English text will produce English output.


class CosyVoiceInference:
    """
    Real CosyVoice 3 inference handler.

    Uses the official CosyVoice AutoModel for inference.
    Supports zero-shot, cross-lingual, and instruct modes.
    """

    # Paths to CosyVoice source code (added to sys.path)
    COSYVOICE_SOURCE_PATH = "/workspace/cosyvoice-official"
    MATCHA_TTS_PATH = "/workspace/cosyvoice-official/third_party/Matcha-TTS"
    DEFAULT_MODEL_PATH = "/workspace/models/Fun-CosyVoice3-0.5B"

    def __init__(self, config: CosyVoiceConfig) -> None:
        """
        Initialize inference handler.

        Args:
            config: CosyVoice configuration
        """
        self.config = config
        self._model: Any = None
        self._loaded: bool = False
        self._sample_rate: int = 22050  # Will be updated from model
        self._registered_speakers: dict[str, str] = {}  # voice_id -> speaker_name

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @property
    def sample_rate(self) -> int:
        """Get the model's output sample rate."""
        return self._sample_rate

    def _setup_paths(self) -> None:
        """Add CosyVoice source paths to sys.path."""
        paths_to_add = [self.COSYVOICE_SOURCE_PATH, self.MATCHA_TTS_PATH]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)
                logger.debug(f"Added to sys.path: {path}")

    async def load_model(self) -> None:
        """
        Load the CosyVoice model using AutoModel.

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info("Loading CosyVoice model...")

        # Setup paths before importing
        self._setup_paths()

        # Determine model path
        model_path = self.config.model_path
        if model_path == "./models/cosyvoice" or not Path(model_path).exists():
            model_path = self.DEFAULT_MODEL_PATH
            logger.info(f"Using default model path: {model_path}")

        # Run loading in thread pool (blocking operation)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_inference_executor, self._load_model_sync, model_path)

        self._loaded = True
        logger.info(f"CosyVoice model loaded successfully (sample_rate={self._sample_rate})")

    def _load_model_sync(self, model_path: str) -> None:
        """Synchronous model loading (runs in thread pool)."""
        try:
            from cosyvoice.cli.cosyvoice import AutoModel

            logger.info(f"Loading model from: {model_path}")

            # Load the model
            self._model = AutoModel(model_dir=model_path)

            # Get sample rate from loaded model
            self._sample_rate = self._model.sample_rate
            logger.info(f"Model sample rate: {self._sample_rate}")

            # Move to appropriate device if needed
            device = self.config.get_device()
            logger.info(f"Using device: {device}")

            # Apply fp16 if configured and CUDA available
            if self.config.dtype == "float16" and torch.cuda.is_available():
                logger.info("Using float16 precision")

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import CosyVoice. Ensure {self.COSYVOICE_SOURCE_PATH} exists "
                f"and contains the cosyvoice package. Error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load CosyVoice model: {e}") from e

    async def unload_model(self) -> None:
        """Unload model and clear CUDA cache."""
        if self._model is not None:
            del self._model
            self._model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        self._registered_speakers.clear()
        logger.info("Model unloaded")

    async def generate(
        self,
        text: str,
        speaker_embedding: Any,
        speed: float = 1.0,
        prompt_audio_path: str | None = None,
        language: str = "en",
    ) -> tuple[torch.Tensor, int]:
        """
        Generate audio from text using inference_zero_shot.

        The output language is determined by the INPUT TEXT language, not by instructions.
        CosyVoice3 will speak whatever language the text is written in.

        Args:
            text: Text to synthesize (the language of this text determines output language)
            speaker_embedding: Speaker embedding (unused, kept for compatibility)
            speed: Speech speed multiplier (0.5-2.0)
            prompt_audio_path: Path to prompt audio file for voice cloning
            language: Language hint (unused - output language is determined by text)

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            _inference_executor,
            self._generate_sync,
            text,
            speaker_embedding,
            speed,
            prompt_audio_path,
            language,
        )

        return audio_tensor, self._sample_rate

    def _generate_sync(
        self,
        text: str,
        speaker_embedding: Any,
        speed: float,
        prompt_audio_path: str | None,
        language: str = "en",
    ) -> torch.Tensor:
        """
        Synchronous generation using inference_zero_shot.
        
        CosyVoice3 auto-detects language from input text - NO language tags needed!
        Just pass plain English text for English output.
        """
        audio_chunks = []

        # If prompt_audio_path is provided and exists, use inference_zero_shot
        if prompt_audio_path and Path(prompt_audio_path).exists():
            logger.info(f"Using zero_shot inference: prompt={prompt_audio_path}")
            logger.info(f"Text (first 100 chars): {text[:100]}...")

            # CosyVoice3 inference_zero_shot: (tts_text, prompt_text, prompt_wav)
            # - tts_text: text to synthesize (language auto-detected from text)
            # - prompt_text: format "You are a helpful assistant.<|endofprompt|>[transcript of prompt audio]"
            # - prompt_wav: path to prompt audio file
            #
            # The prompt_text should contain the transcript of what's spoken in prompt_wav
            # For simplicity, we use a generic prompt since we don't have the transcript
            prompt_text = "You are a helpful assistant.<|endofprompt|>"
            
            for chunk in self._model.inference_zero_shot(
                text,
                prompt_text,
                prompt_audio_path,
                speed=speed,
                stream=False,
            ):
                if "tts_speech" in chunk:
                    audio_chunks.append(chunk["tts_speech"])
        else:
            # No prompt audio available - this model requires reference audio
            raise RuntimeError(
                "This voice requires reference audio for synthesis. "
                "Fun-CosyVoice3 only supports voice cloning mode. "
                "Please clone a voice first using POST /v1/voices/clone with an audio sample, "
                "then use the cloned voice_id for synthesis."
            )

        # Concatenate all chunks
        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=1)
        else:
            # Return empty audio if no chunks
            full_audio = torch.zeros(1, self._sample_rate)

        return full_audio

    async def generate_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str,
        speed: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """
        Generate audio using zero-shot voice cloning.

        Args:
            text: Text to synthesize
            prompt_text: Transcript of the prompt audio
            prompt_audio_path: Path to the prompt audio file
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            _inference_executor,
            self._generate_zero_shot_sync,
            text,
            prompt_text,
            prompt_audio_path,
            speed,
        )

        return audio_tensor, self._sample_rate

    def _generate_zero_shot_sync(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str,
        speed: float,
    ) -> torch.Tensor:
        """Synchronous zero-shot generation (runs in thread pool)."""
        audio_chunks = []

        logger.debug(f"Zero-shot inference: text='{text[:50]}...', prompt='{prompt_text[:30]}...'")

        for chunk in self._model.inference_zero_shot(
            text,
            prompt_text,
            prompt_audio_path,
            speed=speed,
            stream=False,
        ):
            if "tts_speech" in chunk:
                audio_chunks.append(chunk["tts_speech"])

        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=1)
        else:
            full_audio = torch.zeros(1, self._sample_rate)

        return full_audio

    async def generate_instruct(
        self,
        text: str,
        instruction: str,
        prompt_audio_path: str,
        speed: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """
        Generate audio using instruct mode (with instructions like dialect, emotion).

        Args:
            text: Text to synthesize
            instruction: Instructions (e.g., "speak in Cantonese<|endofprompt|>")
            prompt_audio_path: Path to prompt audio file
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        loop = asyncio.get_event_loop()
        audio_tensor = await loop.run_in_executor(
            _inference_executor,
            self._generate_instruct_sync,
            text,
            instruction,
            prompt_audio_path,
            speed,
        )

        return audio_tensor, self._sample_rate

    def _generate_instruct_sync(
        self,
        text: str,
        instruction: str,
        prompt_audio_path: str,
        speed: float,
    ) -> torch.Tensor:
        """Synchronous instruct generation (runs in thread pool)."""
        audio_chunks = []

        logger.debug(f"Instruct inference: instruction='{instruction[:50]}...'")

        for chunk in self._model.inference_instruct2(
            text,
            instruction,
            prompt_audio_path,
            speed=speed,
            stream=False,
        ):
            if "tts_speech" in chunk:
                audio_chunks.append(chunk["tts_speech"])

        if audio_chunks:
            full_audio = torch.cat(audio_chunks, dim=1)
        else:
            full_audio = torch.zeros(1, self._sample_rate)

        return full_audio

    async def extract_speaker_embedding(
        self,
        audio_path: str,
        speaker_name: str | None = None,
    ) -> str:
        """
        Register a speaker from audio and return speaker ID.

        Uses CosyVoice's add_zero_shot_spk() to register the speaker.

        Args:
            audio_path: Path to reference audio file
            speaker_name: Optional name for the speaker

        Returns:
            Speaker ID that can be used for synthesis
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded")

        # Generate unique speaker name if not provided
        if speaker_name is None:
            speaker_name = f"speaker_{uuid.uuid4().hex[:8]}"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _inference_executor,
            self._register_speaker_sync,
            audio_path,
            speaker_name,
        )

        # Store mapping
        self._registered_speakers[speaker_name] = speaker_name
        logger.info(f"Registered speaker: {speaker_name}")

        return speaker_name

    def _register_speaker_sync(self, audio_path: str, speaker_name: str) -> None:
        """Synchronous speaker registration (runs in thread pool)."""
        # CosyVoice3 uses add_zero_shot_spk to register speakers
        if hasattr(self._model, "add_zero_shot_spk"):
            self._model.add_zero_shot_spk(speaker_name, audio_path)
        else:
            logger.warning("add_zero_shot_spk not available, using audio path directly")

    async def _warmup(self) -> None:
        """Run warmup inference to initialize CUDA kernels."""
        logger.info("Running warmup inference...")
        try:
            # Generate a short test phrase
            test_text = "<|en|>Hello world."
            # This will be slow the first time due to CUDA kernel initialization
            # We don't need the result, just warming up
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                _inference_executor,
                lambda: list(self._model.inference_cross_lingual(test_text, "", stream=False)),
            )
            logger.info("Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")


class MockCosyVoiceInference:
    """
    Mock CosyVoice inference for testing without GPU.

    Generates sine wave audio for testing the API pipeline.
    """

    def __init__(self, config: CosyVoiceConfig) -> None:
        """
        Initialize mock inference.

        Args:
            config: CosyVoice configuration
        """
        self.config = config
        self._loaded: bool = False
        self._sample_rate: int = config.sample_rate
        self._registered_speakers: dict[str, str] = {}

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
        logger.info("Loading mock CosyVoice inference...")
        self._loaded = True
        logger.info("Mock CosyVoice inference loaded")

    async def unload_model(self) -> None:
        """Mock model unloading."""
        self._loaded = False
        self._registered_speakers.clear()
        logger.info("Mock inference unloaded")

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
        language: str = "en",
    ) -> tuple[torch.Tensor, int]:
        """
        Generate mock audio (sine wave).

        Args:
            text: Text to synthesize
            speaker_embedding: Ignored for mock
            speed: Speech speed multiplier
            prompt_audio_path: Ignored for mock
            language: Target language (ignored for mock)

        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        audio = self._generate_mock_audio(text, speed)
        return audio, self._sample_rate

    async def generate_zero_shot(
        self,
        text: str,
        prompt_text: str,
        prompt_audio_path: str,
        speed: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """Generate mock audio for zero-shot (same as generate)."""
        audio = self._generate_mock_audio(text, speed)
        return audio, self._sample_rate

    async def generate_instruct(
        self,
        text: str,
        instruction: str,
        prompt_audio_path: str,
        speed: float = 1.0,
    ) -> tuple[torch.Tensor, int]:
        """Generate mock audio for instruct mode (same as generate)."""
        audio = self._generate_mock_audio(text, speed)
        return audio, self._sample_rate

    async def extract_speaker_embedding(
        self,
        audio_path: str,
        speaker_name: str | None = None,
    ) -> str:
        """
        Mock speaker embedding extraction.

        Args:
            audio_path: Path to audio (ignored)
            speaker_name: Optional speaker name

        Returns:
            Speaker ID string
        """
        if speaker_name is None:
            speaker_name = f"mock_speaker_{uuid.uuid4().hex[:8]}"
        self._registered_speakers[speaker_name] = speaker_name
        return speaker_name


def get_inference(config: CosyVoiceConfig) -> CosyVoiceInference | MockCosyVoiceInference:
    """
    Get appropriate inference handler based on config.

    Args:
        config: CosyVoice configuration

    Returns:
        CosyVoiceInference or MockCosyVoiceInference
    """
    if config.use_mock:
        return MockCosyVoiceInference(config)
    return CosyVoiceInference(config)
