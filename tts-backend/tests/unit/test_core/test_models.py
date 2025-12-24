"""Tests for core models."""

from src.core.interfaces import AudioResult, VoiceInfo
from src.core.models import SynthesisRequest, SynthesisResponse
from tests.factories import create_voice_info


class TestVoiceInfo:
    """Tests for VoiceInfo model."""

    def test_voice_info_creation(self):
        """Test basic VoiceInfo creation."""
        voice = VoiceInfo(
            voice_id="test-voice",
            name="Test Voice",
            language="en",
        )

        assert voice.voice_id == "test-voice"
        assert voice.name == "Test Voice"
        assert voice.language == "en"
        assert voice.is_cloned is False

    def test_voice_info_factory(self):
        """Test VoiceInfo creation via factory."""
        voice = create_voice_info(
            voice_id="custom-id",
            name="Custom Name",
            language="zh",
        )

        assert voice.voice_id == "custom-id"
        assert voice.name == "Custom Name"
        assert voice.language == "zh"

    def test_voice_info_with_gender(self):
        """Test VoiceInfo with gender field."""
        voice = VoiceInfo(
            voice_id="test-voice",
            name="Test Voice",
            language="en",
            gender="female",
            description="A test voice",
            is_cloned=True,
        )

        assert voice.gender == "female"
        assert voice.description == "A test voice"
        assert voice.is_cloned is True


class TestAudioResult:
    """Tests for AudioResult model."""

    def test_audio_result_creation(self):
        """Test basic AudioResult creation."""
        result = AudioResult(
            audio_data=b"\x00" * 100,
            sample_rate=22050,
            duration_seconds=1.5,
            format="wav",
        )

        assert len(result.audio_data) == 100
        assert result.sample_rate == 22050
        assert result.duration_seconds == 1.5
        assert result.format == "wav"

    def test_audio_result_with_metadata(self):
        """Test AudioResult with metadata."""
        result = AudioResult(
            audio_data=b"\x00" * 100,
            sample_rate=22050,
            duration_seconds=1.5,
            format="wav",
            metadata={"voice_id": "default", "text_length": 50},
        )

        assert result.metadata["voice_id"] == "default"
        assert result.metadata["text_length"] == 50


class TestSynthesisRequest:
    """Tests for SynthesisRequest model."""

    def test_synthesis_request_creation(self):
        """Test basic SynthesisRequest creation."""
        request = SynthesisRequest(
            text="Hello, world!",
            voice_id="default",
        )

        assert request.text == "Hello, world!"
        assert request.voice_id == "default"
        assert request.speed == 1.0
        assert request.language == "en"


class TestSynthesisResponse:
    """Tests for SynthesisResponse model."""

    def test_synthesis_response_creation(self):
        """Test basic SynthesisResponse creation."""
        response = SynthesisResponse(
            audio_base64="SGVsbG8=",
            format="wav",
            sample_rate=22050,
            duration_seconds=1.5,
            request_id="test-123",
            voice_id="default",
            text_length=13,
            processing_time_ms=50.5,
        )

        assert response.audio_base64 == "SGVsbG8="
        assert response.format == "wav"
        assert response.sample_rate == 22050
        assert response.duration_seconds == 1.5
