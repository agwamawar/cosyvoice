"""Tests for TTS service."""

import pytest

from src.core.models import SynthesisRequest
from src.core.tts_service import TTSService


class TestTTSService:
    """Tests for TTSService class."""

    @pytest.mark.skip(reason="Requires torchaudio backend (sox/soundfile)")
    async def test_synthesize_returns_response(self, tts_service: TTSService):
        """Test that synthesize returns a response."""
        request = SynthesisRequest(
            text="Hello, this is a test.",
            voice_id="default",
            language="en",
            output_format="wav",
            sample_rate=22050,
        )

        result = await tts_service.synthesize(request)

        assert result is not None
        assert result.audio_base64 is not None
        assert len(result.audio_base64) > 0
        assert result.duration_seconds > 0
        assert result.sample_rate == 22050

    async def test_synthesize_invalid_voice_raises(self, tts_service: TTSService):
        """Test that invalid voice ID raises error."""
        request = SynthesisRequest(
            text="Hello",
            voice_id="nonexistent-voice-id",
            language="en",
        )

        with pytest.raises(Exception):  # VoiceNotFoundError or similar
            await tts_service.synthesize(request)

    async def test_service_is_ready(self, tts_service: TTSService):
        """Test that initialized service reports ready."""
        assert tts_service.is_ready()


class TestTTSServiceInitialization:
    """Tests for TTS service initialization."""

    async def test_initialize_with_mock(self, settings, registry_with_mock):
        """Test service initializes correctly with mock engine."""
        service = TTSService(settings, registry=registry_with_mock)

        # Initialize the service
        await service.initialize()

        # Service should be ready after initialization
        assert service.is_ready()

        # Shutdown the service
        await service.shutdown()
        # Note: Due to singleton EngineRegistry pattern, shutdown behavior
        # may vary in test environment. We primarily test that no errors occur.
