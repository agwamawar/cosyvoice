"""Tests for TTS service."""

import pytest

from src.audio.formats import AudioFormat
from src.core.interfaces import SynthesisOptions
from src.core.tts_service import TTSService


class TestTTSService:
    """Tests for TTSService class."""

    async def test_synthesize_returns_audio(self, tts_service: TTSService):
        """Test that synthesize returns audio data."""
        options = SynthesisOptions(
            voice_id="default",
            speed=1.0,
            language="en",
            output_format=AudioFormat.WAV,
            sample_rate=22050,
        )

        result = await tts_service.synthesize(
            text="Hello, this is a test.",
            options=options,
        )

        assert result is not None
        assert result.audio_data is not None
        assert len(result.audio_data) > 0
        assert result.duration_seconds > 0
        assert result.sample_rate == 22050

    async def test_synthesize_invalid_voice_raises(self, tts_service: TTSService):
        """Test that invalid voice ID raises error."""
        options = SynthesisOptions(
            voice_id="nonexistent-voice-id",
            speed=1.0,
            language="en",
            output_format=AudioFormat.WAV,
        )

        with pytest.raises(Exception):  # VoiceNotFoundError or similar
            await tts_service.synthesize(
                text="Hello",
                options=options,
            )

    async def test_list_voices(self, tts_service: TTSService):
        """Test listing available voices."""
        voices = await tts_service.list_voices()

        assert voices is not None
        assert len(voices) > 0

        # Check that default voice exists
        voice_ids = [v.voice_id for v in voices]
        assert "default" in voice_ids

    async def test_get_voice(self, tts_service: TTSService):
        """Test getting a specific voice."""
        voice = await tts_service.get_voice("default")

        assert voice is not None
        assert voice.voice_id == "default"
        assert voice.name is not None

    async def test_get_voice_not_found(self, tts_service: TTSService):
        """Test getting non-existent voice returns None."""
        voice = await tts_service.get_voice("nonexistent-voice")

        assert voice is None

    async def test_service_is_ready(self, tts_service: TTSService):
        """Test that initialized service reports ready."""
        assert tts_service.is_ready()


class TestTTSServiceInitialization:
    """Tests for TTS service initialization."""

    async def test_initialize_with_mock(self, settings):
        """Test service initializes correctly with mock engine."""
        service = TTSService(settings)

        assert not service.is_ready()

        await service.initialize()

        assert service.is_ready()

        await service.shutdown()

        assert not service.is_ready()
