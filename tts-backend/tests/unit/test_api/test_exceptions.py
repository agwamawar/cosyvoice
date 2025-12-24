"""Tests for API exceptions."""

import pytest

from src.api.exceptions import (
    TTSBaseException,
    ValidationError,
    VoiceNotFoundError,
    InvalidAPIKeyError,
    MissingAPIKeyError,
    ModelNotLoadedError,
    SynthesisError,
)


class TestExceptionToResponseFormat:
    """Tests for exception to response format conversion."""

    def test_validation_error_format(self):
        """Test ValidationError converts to proper format."""
        error = ValidationError(
            message="Invalid input",
            details={"field": "text", "reason": "too short"},
        )
        
        response = error.to_response()
        
        assert response["error"]["code"] == error.error_code
        assert response["error"]["message"] == "Invalid input"
        assert "field" in response["error"]["details"]

    def test_voice_not_found_format(self):
        """Test VoiceNotFoundError includes voice_id."""
        error = VoiceNotFoundError("voice-123")
        
        response = error.to_response()
        
        assert "voice-123" in response["error"]["message"]
        assert response["error"]["code"] == "VOICE_NOT_FOUND"

    def test_base_exception_format(self):
        """Test base exception has required fields."""
        error = TTSBaseException(
            message="Something went wrong",
            error_code="GENERIC_ERROR",
        )
        
        response = error.to_response()
        
        assert "error" in response
        assert "code" in response["error"]
        assert "message" in response["error"]


class TestExceptionStatusCodes:
    """Tests for exception HTTP status codes."""

    def test_validation_error_status_code(self):
        """Test ValidationError has 400 status code."""
        error = ValidationError("Bad request")
        assert error.status_code == 400

    def test_missing_api_key_status_code(self):
        """Test MissingAPIKeyError has 401 status code."""
        error = MissingAPIKeyError()
        assert error.status_code == 401

    def test_invalid_api_key_status_code(self):
        """Test InvalidAPIKeyError has 401 status code."""
        error = InvalidAPIKeyError()
        assert error.status_code == 401

    def test_voice_not_found_status_code(self):
        """Test VoiceNotFoundError has 404 status code."""
        error = VoiceNotFoundError("voice-123")
        assert error.status_code == 404

    def test_model_not_loaded_status_code(self):
        """Test ModelNotLoadedError has 503 status code."""
        error = ModelNotLoadedError()
        assert error.status_code == 503

    def test_synthesis_error_status_code(self):
        """Test SynthesisError has 500 status code."""
        error = SynthesisError("Synthesis failed")
        assert error.status_code == 500


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

    def test_error_codes_are_uppercase(self):
        """Test that error codes follow naming convention."""
        exceptions = [
            ValidationError("test"),
            VoiceNotFoundError("test"),
            InvalidAPIKeyError(),
        ]
        
        for exc in exceptions:
            assert exc.error_code == exc.error_code.upper()
            assert "_" in exc.error_code or exc.error_code.isalpha()
