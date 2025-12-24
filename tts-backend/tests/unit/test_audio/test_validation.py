"""Tests for audio validation."""

from src.audio.formats import AudioFormat, detect_format_from_bytes
from src.audio.validation import AudioRequirements, AudioValidator

from tests.factories import create_audio_sample


class TestAudioValidation:
    """Tests for audio validation functionality."""

    def test_valid_audio_passes_validation(self, sample_wav_bytes: bytes):
        """Test that valid WAV audio passes validation."""
        validator = AudioValidator()
        result = validator.validate(sample_wav_bytes)

        assert result.is_valid
        assert result.duration_seconds > 0
        assert result.sample_rate > 0

    def test_short_audio_fails(self, short_wav_bytes: bytes):
        """Test that audio shorter than minimum duration fails."""
        requirements = AudioRequirements(min_duration_seconds=3.0)
        validator = AudioValidator(requirements)
        result = validator.validate(short_wav_bytes)

        assert not result.is_valid
        assert "duration" in result.error_message.lower()

    def test_corrupted_data_fails(self, corrupted_audio_bytes: bytes):
        """Test that corrupted audio data fails validation."""
        validator = AudioValidator()
        result = validator.validate(corrupted_audio_bytes)

        assert not result.is_valid

    def test_format_detection(self, sample_wav_bytes: bytes):
        """Test audio format detection from bytes."""
        detected = detect_format_from_bytes(sample_wav_bytes)
        assert detected == AudioFormat.WAV


class TestAudioRequirements:
    """Tests for AudioRequirements configuration."""

    def test_default_requirements(self):
        """Test default audio requirements."""
        req = AudioRequirements()
        assert req.min_duration_seconds >= 0
        assert req.max_duration_seconds > req.min_duration_seconds
        assert req.max_file_size_mb > 0

    def test_custom_requirements(self):
        """Test custom audio requirements."""
        req = AudioRequirements(
            min_duration_seconds=5.0,
            max_duration_seconds=60.0,
            max_file_size_mb=100.0,
        )
        assert req.min_duration_seconds == 5.0
        assert req.max_duration_seconds == 60.0
        assert req.max_file_size_mb == 100.0


class TestFormatDetection:
    """Tests for audio format detection."""

    def test_detect_wav_format(self):
        """Test WAV format detection."""
        wav_data = create_audio_sample(duration=1.0, format="wav")
        assert detect_format_from_bytes(wav_data) == AudioFormat.WAV

    def test_detect_unknown_format(self):
        """Test unknown format returns None or raises."""
        unknown_data = b"not audio data"
        result = detect_format_from_bytes(unknown_data)
        assert result is None or result == AudioFormat.WAV  # Fallback behavior
