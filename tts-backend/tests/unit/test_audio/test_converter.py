"""Tests for audio converter."""

import pytest

from src.audio.converter import AudioConverter, ConversionOptions
from src.audio.formats import AudioFormat
from tests.factories import create_audio_sample


class TestAudioConverter:
    """Tests for AudioConverter class."""

    @pytest.fixture
    def converter(self) -> AudioConverter:
        """Create audio converter instance."""
        return AudioConverter()

    async def test_convert_with_options(self, converter: AudioConverter):
        """Test converting audio with ConversionOptions."""
        wav_data = create_audio_sample(duration=1.0, format="wav")
        options = ConversionOptions(
            target_format=AudioFormat.WAV,
            sample_rate=22050,
            channels=1,
        )

        result = await converter.convert(
            audio_data=wav_data,
            source_format=AudioFormat.WAV,
            options=options,
        )

        assert result is not None
        assert len(result) > 0

    async def test_resample_audio(self, converter: AudioConverter):
        """Test resampling audio to different sample rate."""
        wav_data = create_audio_sample(duration=1.0, sample_rate=44100)

        result = await converter.resample(
            audio_data=wav_data,
            source_rate=44100,
            target_rate=22050,
        )

        assert result is not None
        assert len(result) > 0


class TestConversionOptions:
    """Tests for ConversionOptions model."""

    def test_default_options(self):
        """Test default conversion options."""
        options = ConversionOptions()

        assert options.target_format == AudioFormat.WAV
        assert options.sample_rate == 22050
        assert options.channels == 1
        assert options.normalize is True

    def test_custom_options(self):
        """Test custom conversion options."""
        options = ConversionOptions(
            target_format=AudioFormat.MP3,
            sample_rate=44100,
            channels=2,
            normalize=False,
        )

        assert options.target_format == AudioFormat.MP3
        assert options.sample_rate == 44100
        assert options.channels == 2
        assert options.normalize is False
