"""Tests for API request/response models."""

import pytest
from pydantic import ValidationError

from src.api.models.requests import SynthesisRequest, VoiceUploadRequest
from src.api.models.responses import HealthResponse, SynthesisResponse
from src.config.settings import AudioFormat


class TestSynthesisRequest:
    """Tests for SynthesisRequest model."""

    def test_valid_request(self):
        """Test valid synthesis request."""
        request = SynthesisRequest(
            text="Hello, world!",
            voice_id="default",
            language="en",
        )

        assert request.text == "Hello, world!"
        assert request.voice_id == "default"
        assert request.language == "en"
        assert request.output_format == AudioFormat.WAV
        assert request.speed == 1.0
        assert request.pitch == 1.0

    def test_text_whitespace_stripped(self):
        """Test that text whitespace is stripped."""
        request = SynthesisRequest(
            text="  Hello, world!  ",
        )

        assert request.text == "Hello, world!"

    def test_empty_text_rejected(self):
        """Test that empty text is rejected."""
        with pytest.raises(ValidationError):
            SynthesisRequest(text="")

    def test_whitespace_only_text_rejected(self):
        """Test that whitespace-only text is rejected."""
        with pytest.raises(ValidationError):
            SynthesisRequest(text="   ")

    def test_text_too_long_rejected(self):
        """Test that too-long text is rejected."""
        with pytest.raises(ValidationError):
            SynthesisRequest(text="x" * 5001)

    def test_speed_bounds(self):
        """Test speed parameter bounds."""
        # Valid speeds
        SynthesisRequest(text="test", speed=0.5)
        SynthesisRequest(text="test", speed=2.0)

        # Invalid speeds
        with pytest.raises(ValidationError):
            SynthesisRequest(text="test", speed=0.4)

        with pytest.raises(ValidationError):
            SynthesisRequest(text="test", speed=2.1)

    def test_pitch_bounds(self):
        """Test pitch parameter bounds."""
        # Valid pitches
        SynthesisRequest(text="test", pitch=0.5)
        SynthesisRequest(text="test", pitch=2.0)

        # Invalid pitches
        with pytest.raises(ValidationError):
            SynthesisRequest(text="test", pitch=0.4)

        with pytest.raises(ValidationError):
            SynthesisRequest(text="test", pitch=2.1)


class TestVoiceUploadRequest:
    """Tests for VoiceUploadRequest model."""

    def test_valid_request(self):
        """Test valid voice upload request."""
        request = VoiceUploadRequest(
            name="My Voice",
            description="A custom voice",
            language="en",
        )

        assert request.name == "My Voice"
        assert request.description == "A custom voice"
        assert request.language == "en"

    def test_name_stripped(self):
        """Test that name whitespace is stripped."""
        request = VoiceUploadRequest(name="  My Voice  ")
        assert request.name == "My Voice"

    def test_empty_name_rejected(self):
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError):
            VoiceUploadRequest(name="")


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_health_response(self):
        """Test health response creation."""
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            environment="dev",
            checks={"model_loaded": True},
        )

        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.environment == "dev"
        assert response.checks["model_loaded"] is True
        assert response.timestamp is not None


class TestSynthesisResponse:
    """Tests for SynthesisResponse model."""

    def test_synthesis_response(self):
        """Test synthesis response creation."""
        response = SynthesisResponse(
            audio_base64="SGVsbG8=",
            duration_seconds=1.5,
            sample_rate=22050,
            format="wav",
            voice_id="default",
            text_length=100,
            processing_time_ms=50.5,
        )

        assert response.audio_base64 == "SGVsbG8="
        assert response.duration_seconds == 1.5
        assert response.sample_rate == 22050
        assert response.format == "wav"
        assert response.voice_id == "default"
        assert response.text_length == 100
        assert response.processing_time_ms == 50.5
