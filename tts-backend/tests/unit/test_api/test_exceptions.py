"""Tests for API exceptions."""

from src.api.exceptions import (
    InvalidAPIKeyError,
    MissingAPIKeyError,
    ModelNotLoadedError,
    SynthesisError,
    TTSBaseException,
    ValidationError,
    VoiceNotFoundError,
)


class TestExceptionToResponseFormat:
    """Tests for exception to dict format conversion."""

    def test_validation_error_format(self):
        """Test ValidationError converts to proper format."""
        error = ValidationError(
            message="Invalid input",
            details={"field": "text", "reason": "too short"},
        )

        response = error.to_dict()

        assert response["code"] == error.error_code
        assert response["message"] == "Invalid input"
        assert "field" in response["details"]

    def test_voice_not_found_format(self):
        """Test VoiceNotFoundError includes voice_id."""
        error = VoiceNotFoundError("voice-123")

        response = error.to_dict()

        assert "voice-123" in response["message"]
        assert response["code"] == "TTS_E301"

    def test_base_exception_format(self):
        """Test base exception has required fields."""
        error = TTSBaseException(
            message="Something went wrong",
            error_code="GENERIC_ERROR",
        )

        response = error.to_dict()

        assert "code" in response
        assert "message" in response
        assert "details" in response


class TestExceptionStatusCodes:
    """Tests for exception HTTP status codes."""

    def test_validation_error_http_status(self):
        """Test ValidationError has 400 http_status."""
        error = ValidationError("Bad request")
        assert error.http_status == 400

    def test_missing_api_key_http_status(self):
        """Test MissingAPIKeyError has 401 http_status."""
        error = MissingAPIKeyError()
        assert error.http_status == 401

    def test_invalid_api_key_http_status(self):
        """Test InvalidAPIKeyError has 401 http_status."""
        error = InvalidAPIKeyError()
        assert error.http_status == 401

    def test_voice_not_found_http_status(self):
        """Test VoiceNotFoundError has 404 http_status."""
        error = VoiceNotFoundError("voice-123")
        assert error.http_status == 404

    def test_model_not_loaded_http_status(self):
        """Test ModelNotLoadedError has 503 http_status."""
        error = ModelNotLoadedError()
        assert error.http_status == 503

    def test_synthesis_error_http_status(self):
        """Test SynthesisError has 500 http_status."""
        error = SynthesisError("Synthesis failed")
        assert error.http_status == 500


class TestExceptionErrorCodes:
    """Tests for exception error codes."""

    def test_unique_error_codes(self):
        """Test that each exception type has a unique error code."""
        exceptions = [
            ValidationError("test"),
            VoiceNotFoundError("test"),
            InvalidAPIKeyError(),
            MissingAPIKeyError(),
            ModelNotLoadedError(),
            SynthesisError("test"),
        ]

        codes = [e.error_code for e in exceptions]

        # All codes should be unique
        assert len(codes) == len(set(codes))

    def test_error_codes_follow_convention(self):
        """Test that error codes follow TTS_EXXX naming convention."""
        exceptions = [
            ValidationError("test"),
            VoiceNotFoundError("test"),
            InvalidAPIKeyError(),
        ]

        for exc in exceptions:
            assert exc.error_code.startswith("TTS_E")
