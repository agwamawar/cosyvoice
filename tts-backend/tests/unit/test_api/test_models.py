"""Tests for API request/response models."""

import pytest
from pydantic import ValidationError

from src.api.models.requests import CloneVoiceRequest, SynthesizeRequest
from src.api.models.responses import HealthResponse, SynthesizeResponse


class TestSynthesizeRequest:
    """Tests for SynthesizeRequest model."""

    def test_valid_request(self):
        """Test valid synthesis request."""
        request = SynthesizeRequest(
            text="Hello, world!",
            voice_id="default",
            language="en",
        )

        assert request.text == "Hello, world!"
        assert request.voice_id == "default"
        assert request.language == "en"
        assert request.output_format == "wav"
        assert request.speed == 1.0

    def test_text_whitespace_stripped(self):
        """Test that text whitespace is stripped."""
        request = SynthesizeRequest(
            text="  Hello, world!  ",
        )

        assert request.text == "Hello, world!"

    def test_empty_text_rejected(self):
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            SynthesizeRequest(text="")

    def test_whitespace_only_text_rejected(self):
        """Test that whitespace-only text is rejected."""
        with pytest.raises(ValidationError):
            SynthesizeRequest(text="   ")

    def test_text_too_long_rejected(self):
        """Test that too-long text is rejected."""
        with pytest.raises(ValidationError):
            SynthesizeRequest(text="x" * 5001)

    def test_speed_bounds(self):
        """Test speed parameter bounds."""
        # Valid speeds
        SynthesizeRequest(text="test", speed=0.5)
        SynthesizeRequest(text="test", speed=2.0)

        # Invalid speeds
        with pytest.raises(ValidationError):
            SynthesizeRequest(text="test", speed=0.4)

        with pytest.raises(ValidationError):
            SynthesizeRequest(text="test", speed=2.1)


class TestCloneVoiceRequest:
    """Tests for CloneVoiceRequest model."""

    def test_valid_request(self):
        """Test valid voice clone request."""
        request = CloneVoiceRequest(
            voice_name="My Voice",
            description="A custom voice",
            language="en",
        )

        assert request.voice_name == "My Voice"
        assert request.description == "A custom voice"
        assert request.language == "en"

    def test_name_stripped(self):
        """Test that voice_name whitespace is stripped."""
        request = CloneVoiceRequest(voice_name="  My Voice  ")
        assert request.voice_name == "My Voice"

    def test_empty_name_rejected(self):
        """Test that empty voice_name is rejected."""
        with pytest.raises(ValidationError):
            CloneVoiceRequest(voice_name="")


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self):
        """Test health response creation."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.timestamp is not None


class TestSynthesizeResponse:
    """Tests for SynthesizeResponse model."""

    def test_synthesis_response(self):
        """Test synthesis response creation."""
        response = SynthesizeResponse(
            audio_base64="SGVsbG8=",
            duration_seconds=1.5,
            sample_rate=22050,
            format="wav",
            voice_id="default",
            text_length=100,
            processing_time_ms=50.5,
            request_id="test-123",
        )

        assert response.audio_base64 == "SGVsbG8="
        assert response.duration_seconds == 1.5
        assert response.sample_rate == 22050
        assert response.format == "wav"
        assert response.voice_id == "default"
        assert response.text_length == 100
        assert response.processing_time_ms == 50.5
        assert response.request_id == "test-123"
