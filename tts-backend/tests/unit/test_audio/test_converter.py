"""Tests for audio converter."""

import pytest

from src.audio.converter import AudioConverter
from src.audio.formats import AudioFormat
from tests.factories import create_audio_sample


class TestAudioConverter:
    """Tests for AudioConverter class."""

    @pytest.fixture
    def converter(self) -> AudioConverter:
        """Create audio converter instance."""
        return AudioConverter()

    async def test_wav_to_mp3_conversion(self, converter: AudioConverter):
        """Test converting WAV to MP3 format."""
        wav_data = create_audio_sample(duration=1.0, format="wav")
        
        result = await converter.convert(
            audio_data=wav_data,
            source_format=AudioFormat.WAV,
            target_format=AudioFormat.MP3,
        )
        
        assert result is not None
        assert len(result) > 0
        # MP3 files start with specific bytes or ID3 tag
        assert result[:3] == b"ID3" or result[:2] == b"\xff\xfb"

    async def test_resample_audio(self, converter: AudioConverter):
        """Test resampling audio to different sample rate."""
        wav_data = create_audio_sample(duration=1.0, sample_rate=44100)
        
        result = await converter.resample(
            audio_data=wav_data,
            target_sample_rate=22050,
        )
        
        assert result is not None
        assert len(result) > 0

    async def test_normalize_audio(self, converter: AudioConverter):
        """Test normalizing audio volume."""
        wav_data = create_audio_sample(duration=1.0)
        
        result = await converter.normalize(
            audio_data=wav_data,
            target_db=-3.0,
        )
        
        assert result is not None
        assert len(result) > 0

    async def test_convert_to_mono(self, converter: AudioConverter):
        """Test converting stereo to mono."""
        wav_data = create_audio_sample(duration=1.0)
        
        result = await converter.to_mono(audio_data=wav_data)
        
        assert result is not None
        assert len(result) > 0


class TestConversionOptions:
    """Tests for conversion options."""

    def test_supported_formats(self):
        """Test that common formats are supported."""
        converter = AudioConverter()
        supported = converter.supported_formats
        
        assert AudioFormat.WAV in supported
        assert AudioFormat.MP3 in supported

    def test_supported_sample_rates(self):
        """Test that common sample rates are supported."""
        converter = AudioConverter()
        rates = converter.supported_sample_rates
        
        assert 22050 in rates
        assert 44100 in rates
        assert 48000 in rates
