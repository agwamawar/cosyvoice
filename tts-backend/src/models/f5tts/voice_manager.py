"""Voice management for F5-TTS.

Provides:
- VoiceManager class for managing voice profiles
- File operations for saving/loading voice data
- Default voice entries
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.core.interfaces import VoiceInfo
from src.utils.logging import get_logger

logger = get_logger("inference")

# Bundled reference audio files
BACKEND_VOICES_PATH = Path("/workspace/cosyvoice/tts-backend/voices")

# Default reference audio for built-in voices
DEFAULT_REFERENCE_AUDIO = {
    "default": BACKEND_VOICES_PATH / "english_prompt.wav",
    "en-female-1": BACKEND_VOICES_PATH / "english_prompt.wav",
    "en-male-1": BACKEND_VOICES_PATH / "english_prompt.wav",
}

# Default reference text (transcript of the reference audio)
DEFAULT_REFERENCE_TEXT = {
    "default": "",
    "en-female-1": "",
    "en-male-1": "",
}

# Default built-in voices
DEFAULT_VOICES = [
    VoiceInfo(
        voice_id="default",
        name="Default Voice",
        language="en",
        gender="neutral",
        is_cloned=False,
        description="Default English voice",
        tags=["default", "english"],
    ),
    VoiceInfo(
        voice_id="en-female-1",
        name="Emma",
        language="en",
        gender="female",
        is_cloned=False,
        description="Natural English female voice",
        tags=["english", "female", "natural"],
    ),
    VoiceInfo(
        voice_id="en-male-1",
        name="James",
        language="en",
        gender="male",
        is_cloned=False,
        description="Natural English male voice",
        tags=["english", "male", "natural"],
    ),
]


class VoiceManager:
    """
    Manages voice profiles for F5-TTS.

    F5-TTS uses reference audio directly for voice cloning,
    so we store the audio file path and transcript.
    """

    def __init__(
        self,
        voices_directory: str,
        embedding_dim: int = 192,
        device: str = "cpu",
    ) -> None:
        """
        Initialize voice manager.

        Args:
            voices_directory: Directory containing voice profiles
            embedding_dim: Dimension of speaker embeddings (unused for F5-TTS)
            device: Device for tensors
        """
        self.voices_dir = Path(voices_directory)
        self.embedding_dim = embedding_dim
        self.device = device
        self._voices: dict[str, VoiceInfo] = {}
        self._reference_texts: dict[str, str] = {}
        self._loaded = False

    async def load_voices(self) -> None:
        """
        Load all voices from the voices directory.

        Includes default built-in voices and any custom cloned voices.
        """
        # Load default voices first
        for voice in DEFAULT_VOICES:
            self._voices[voice.voice_id] = voice
            # Store default reference texts
            self._reference_texts[voice.voice_id] = DEFAULT_REFERENCE_TEXT.get(
                voice.voice_id, ""
            )

        logger.info(f"Loaded {len(DEFAULT_VOICES)} default voices")

        # Create voices directory if it doesn't exist
        self.voices_dir.mkdir(parents=True, exist_ok=True)

        # Load custom voices from directory
        custom_count = 0
        for voice_dir in self.voices_dir.iterdir():
            if voice_dir.is_dir():
                try:
                    voice = await self._load_voice_from_dir(voice_dir)
                    if voice:
                        self._voices[voice.voice_id] = voice
                        custom_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load voice from {voice_dir}: {e}")

        if custom_count > 0:
            logger.info(f"Loaded {custom_count} custom voices")

        self._loaded = True

    async def _load_voice_from_dir(self, voice_dir: Path) -> VoiceInfo | None:
        """Load a voice from a directory."""
        config_path = voice_dir / "config.json"

        if not config_path.exists():
            return None

        with open(config_path) as f:
            config = json.load(f)

        voice_id = config.get("id", voice_dir.name)

        # Load reference text if exists
        ref_text = config.get("reference_text", "")
        self._reference_texts[voice_id] = ref_text

        return VoiceInfo(
            voice_id=voice_id,
            name=config.get("name", voice_dir.name),
            language=config.get("language", "en"),
            gender=config.get("gender"),
            is_cloned=True,
            description=config.get("description"),
            tags=config.get("tags", []),
        )

    async def get_voice(self, voice_id: str) -> VoiceInfo | None:
        """
        Get voice info by ID.

        Args:
            voice_id: Voice identifier

        Returns:
            VoiceInfo or None if not found
        """
        return self._voices.get(voice_id)

    async def get_embedding(self, voice_id: str) -> Any:
        """
        Get speaker embedding for a voice.

        F5-TTS doesn't use embeddings, but we return a placeholder
        for compatibility with the engine interface.

        Args:
            voice_id: Voice identifier

        Returns:
            Placeholder tensor
        """
        return torch.zeros(self.embedding_dim, device=self.device)

    async def get_reference_text(self, voice_id: str) -> str:
        """
        Get the reference text (transcript) for a voice.

        Args:
            voice_id: Voice identifier

        Returns:
            Reference text string
        """
        return self._reference_texts.get(voice_id, "")

    async def save_voice(
        self,
        voice_name: str,
        embedding: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Save a new voice.

        Args:
            voice_name: Display name for the voice
            embedding: Speaker embedding (unused for F5-TTS)
            metadata: Additional metadata including reference_text

        Returns:
            Generated voice_id
        """
        # Generate unique voice ID
        voice_id = f"custom-{uuid.uuid4().hex[:8]}"

        # Create voice directory
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        # Extract reference text from metadata
        ref_text = metadata.get("reference_text", "") if metadata else ""
        self._reference_texts[voice_id] = ref_text

        # Prepare config
        config = {
            "id": voice_id,
            "name": voice_name,
            "language": metadata.get("language", "en") if metadata else "en",
            "gender": metadata.get("gender") if metadata else None,
            "description": metadata.get("description") if metadata else None,
            "tags": metadata.get("tags", []) if metadata else [],
            "reference_text": ref_text,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Save config
        config_path = voice_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Create VoiceInfo
        voice = VoiceInfo(
            voice_id=voice_id,
            name=voice_name,
            language=config["language"],
            gender=config.get("gender"),
            is_cloned=True,
            description=config.get("description"),
            tags=config.get("tags", []),
        )

        # Add to cache
        self._voices[voice_id] = voice

        logger.info(f"Saved new voice: {voice_id} ({voice_name})")
        return voice_id

    async def delete_voice(self, voice_id: str) -> None:
        """
        Delete a voice.

        Args:
            voice_id: Voice identifier to delete

        Raises:
            ValueError: If voice is a built-in voice or doesn't exist
        """
        if voice_id not in self._voices:
            raise ValueError(f"Voice not found: {voice_id}")

        voice = self._voices[voice_id]
        if not voice.is_cloned:
            raise ValueError(f"Cannot delete built-in voice: {voice_id}")

        # Remove from cache
        del self._voices[voice_id]
        if voice_id in self._reference_texts:
            del self._reference_texts[voice_id]

        # Remove from disk
        voice_dir = self.voices_dir / voice_id
        if voice_dir.exists():
            shutil.rmtree(voice_dir)

        logger.info(f"Deleted voice: {voice_id}")

    async def list_voices(self) -> list[VoiceInfo]:
        """
        List all available voices.

        Returns:
            List of VoiceInfo objects
        """
        return list(self._voices.values())

    async def save_reference_audio(
        self,
        voice_id: str,
        audio_data: bytes,
        filename: str = "reference.wav",
    ) -> Path:
        """
        Save reference audio for a voice.

        Args:
            voice_id: Voice identifier
            audio_data: Raw audio bytes
            filename: Filename for the audio

        Returns:
            Path to saved audio file
        """
        voice_dir = self.voices_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)

        audio_path = voice_dir / filename
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        return audio_path

    async def get_reference_audio_path(self, voice_id: str) -> Path | None:
        """
        Get the path to reference audio for a voice.

        Args:
            voice_id: Voice identifier

        Returns:
            Path to reference audio file, or None if not found
        """
        # First check custom voice directory
        voice_dir = self.voices_dir / voice_id
        reference_path = voice_dir / "reference.wav"

        if reference_path.exists():
            return reference_path

        # Check for other audio formats in custom directory
        for ext in [".wav", ".mp3", ".flac", ".ogg"]:
            audio_path = voice_dir / f"reference{ext}"
            if audio_path.exists():
                return audio_path

        # Check bundled reference audio for default voices
        if voice_id in DEFAULT_REFERENCE_AUDIO:
            bundled_path = DEFAULT_REFERENCE_AUDIO[voice_id]
            if bundled_path.exists():
                logger.debug(f"Using bundled reference audio for {voice_id}: {bundled_path}")
                return bundled_path

        return None

    async def update_reference_text(self, voice_id: str, reference_text: str) -> None:
        """
        Update the reference text for a voice.

        Args:
            voice_id: Voice identifier
            reference_text: New reference text
        """
        self._reference_texts[voice_id] = reference_text

        # Update config file if it's a custom voice
        voice_dir = self.voices_dir / voice_id
        config_path = voice_dir / "config.json"

        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            config["reference_text"] = reference_text

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

    @property
    def is_loaded(self) -> bool:
        """Check if voices have been loaded."""
        return self._loaded
