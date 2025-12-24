"""Tests for core models."""

from datetime import datetime

import pytest

from src.core.models import AudioResult, Voice
from tests.factories import create_audio_result, create_voice


class TestVoice:
    """Tests for Voice model."""

    def test_voice_creation(self):
        """Test basic voice creation."""
        voice = Voice(
            id="test-voice",
            name="Test Voice",
            language="en",
        )

        assert voice.id == "test-voice"
        assert voice.name == "Test Voice"
        assert voice.language == "en"
        assert voice.is_custom is False

    def test_voice_factory(self):
        """Test voice creation via factory."""
        voice = create_voice(
            id="custom-id",
            name="Custom Name",
            language="zh",
        )

        assert voice.id == "custom-id"
        assert voice.name == "Custom Name"
        assert voice.language == "zh"

    def test_voice_to_dict(self):
        """Test voice serialization to dictionary."""
        voice = Voice(
            id="test-voice",
            name="Test Voice",
            language="en",
            description="A test voice",
            gender="female",
            tags=["test", "english"],
            is_custom=True,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
        )

        result = voice.to_dict()

        assert result["id"] == "test-voice"
        assert result["name"] == "Test Voice"
        assert result["language"] == "en"
        assert result["description"] == "A test voice"
        assert result["gender"] == "female"
        assert result["tags"] == ["test", "english"]
        assert result["is_custom"] is True
        assert result["created_at"] == "2024-01-01T12:00:00"

    def test_voice_to_dict_none_created_at(self):
        """Test voice serialization with None created_at."""
        voice = Voice(
            id="test-voice",
            name="Test Voice",
            language="en",
            created_at=None,
        )

        result = voice.to_dict()
        assert result["created_at"] is None


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
        assert result.channels == 1  # default

    def test_audio_result_factory(self):
        """Test AudioResult creation via factory."""
        result = create_audio_result(
            duration_seconds=2.5,
            sample_rate=44100,
        )

        assert result.duration_seconds == 2.5
        assert result.sample_rate == 44100
