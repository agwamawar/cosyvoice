"""Custom exceptions for the TTS API.

Exception hierarchy with unique error codes:
- TTSBaseException: Base with error_code, message, details, http_status
- ValidationError (400): Invalid input, bad parameters
- AuthenticationError (401): Invalid or missing API key
- NotFoundError (404): Voice not found, resource missing
- RateLimitError (429): Too many requests
- ModelError (500): Inference failed, model issues
- AudioProcessingError (500): Encoding/decoding failures
- ResourceError (503): GPU unavailable, out of memory
"""

from typing import Any


class TTSBaseException(Exception):
    """
    Base exception for all TTS API errors.

    All exceptions have:
    - error_code: Unique string identifier (e.g., "TTS_E001")
    - message: Human-readable error message
    - details: Additional context as a dictionary
    - http_status: HTTP status code for the response
    """

    error_code: str = "TTS_E000"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        error_code: str | None = None,
        http_status: int | None = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        if error_code is not None:
            self.error_code = error_code
        if http_status is not None:
            self.http_status = http_status
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            "code": self.error_code,
            "message": self.message,
            "details": self.details,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, message={self.message!r})"


# =============================================================================
# 400 Bad Request - Validation Errors
# =============================================================================


class ValidationError(TTSBaseException):
    """Raised when request validation fails (400)."""

    error_code = "TTS_E100"
    http_status = 400

    def __init__(
        self,
        message: str = "Validation failed",
        details: dict[str, Any] | None = None,
        field: str | None = None,
    ) -> None:
        if field:
            details = details or {}
            details["field"] = field
        super().__init__(message=message, details=details)


class InvalidInputError(ValidationError):
    """Raised when input data is invalid (400)."""

    error_code = "TTS_E101"

    def __init__(
        self,
        message: str = "Invalid input data",
        field: str | None = None,
        value: Any = None,
    ) -> None:
        details: dict[str, Any] = {}
        if field:
            details["field"] = field
        if value is not None:
            details["received_value"] = str(value)[:100]  # Truncate for safety
        super().__init__(message=message, details=details)


class TextTooLongError(ValidationError):
    """Raised when input text exceeds maximum length (400)."""

    error_code = "TTS_E102"

    def __init__(self, max_length: int, actual_length: int) -> None:
        super().__init__(
            message=f"Text exceeds maximum length of {max_length} characters",
            details={"max_length": max_length, "actual_length": actual_length},
        )


class InvalidAudioFormatError(ValidationError):
    """Raised when audio format is invalid or unsupported (400)."""

    error_code = "TTS_E103"

    def __init__(self, format: str, supported_formats: list[str] | None = None) -> None:
        details: dict[str, Any] = {"format": format}
        if supported_formats:
            details["supported_formats"] = supported_formats
        super().__init__(
            message=f"Unsupported audio format: {format}",
            details=details,
        )


class InvalidParameterError(ValidationError):
    """Raised when a parameter value is out of allowed range (400)."""

    error_code = "TTS_E104"

    def __init__(
        self,
        parameter: str,
        value: Any,
        min_value: Any = None,
        max_value: Any = None,
    ) -> None:
        message = f"Invalid value for parameter '{parameter}': {value}"
        details: dict[str, Any] = {"parameter": parameter, "value": value}
        if min_value is not None:
            details["min_value"] = min_value
        if max_value is not None:
            details["max_value"] = max_value
        super().__init__(message=message, details=details)


# =============================================================================
# 401 Unauthorized - Authentication Errors
# =============================================================================


class AuthenticationError(TTSBaseException):
    """Raised when authentication fails (401)."""

    error_code = "TTS_E200"
    http_status = 401

    def __init__(
        self,
        message: str = "Authentication required",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, details=details)


class InvalidAPIKeyError(AuthenticationError):
    """Raised when API key is invalid (401)."""

    error_code = "TTS_E201"

    def __init__(self) -> None:
        super().__init__(message="Invalid API key")


class MissingAPIKeyError(AuthenticationError):
    """Raised when API key is missing (401)."""

    error_code = "TTS_E202"

    def __init__(self) -> None:
        super().__init__(
            message="API key is required",
            details={"header": "X-API-Key"},
        )


class ExpiredAPIKeyError(AuthenticationError):
    """Raised when API key has expired (401)."""

    error_code = "TTS_E203"

    def __init__(self) -> None:
        super().__init__(message="API key has expired")


# =============================================================================
# 404 Not Found - Resource Errors
# =============================================================================


class NotFoundError(TTSBaseException):
    """Raised when a requested resource is not found (404)."""

    error_code = "TTS_E300"
    http_status = 404

    def __init__(
        self,
        resource: str,
        identifier: str,
        message: str | None = None,
    ) -> None:
        super().__init__(
            message=message or f"{resource} not found: {identifier}",
            details={"resource": resource, "identifier": identifier},
        )


class VoiceNotFoundError(NotFoundError):
    """Raised when a voice is not found (404)."""

    error_code = "TTS_E301"

    def __init__(self, voice_id: str) -> None:
        super().__init__(resource="Voice", identifier=voice_id)


class ModelNotFoundError(NotFoundError):
    """Raised when a model file is not found (404)."""

    error_code = "TTS_E302"

    def __init__(self, model_path: str) -> None:
        super().__init__(
            resource="Model",
            identifier=model_path,
            message=f"Model not found at path: {model_path}",
        )


# =============================================================================
# 429 Too Many Requests - Rate Limiting
# =============================================================================


class RateLimitError(TTSBaseException):
    """Raised when rate limit is exceeded (429)."""

    error_code = "TTS_E400"
    http_status = 429

    def __init__(
        self,
        retry_after: int = 60,
        limit: int | None = None,
        window: int | None = None,
    ) -> None:
        details: dict[str, Any] = {"retry_after_seconds": retry_after}
        if limit is not None:
            details["limit"] = limit
        if window is not None:
            details["window_seconds"] = window
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds.",
            details=details,
        )
        self.retry_after = retry_after


# =============================================================================
# 500 Internal Server Error - Model/Processing Errors
# =============================================================================


class ModelError(TTSBaseException):
    """Raised when model inference fails (500)."""

    error_code = "TTS_E500"
    http_status = 500

    def __init__(
        self,
        message: str = "Model inference failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message=message, details=details)


class ModelLoadError(ModelError):
    """Raised when model fails to load (500)."""

    error_code = "TTS_E501"

    def __init__(self, model_name: str, reason: str | None = None) -> None:
        details: dict[str, Any] = {"model": model_name}
        if reason:
            details["reason"] = reason
        super().__init__(
            message=f"Failed to load model: {model_name}",
            details=details,
        )


class InferenceError(ModelError):
    """Raised when model inference fails (500)."""

    error_code = "TTS_E502"

    def __init__(self, message: str = "Inference failed", reason: str | None = None) -> None:
        details: dict[str, Any] = {}
        if reason:
            details["reason"] = reason
        super().__init__(message=message, details=details)


class AudioProcessingError(TTSBaseException):
    """Raised when audio processing fails (500)."""

    error_code = "TTS_E510"
    http_status = 500

    def __init__(
        self,
        message: str = "Audio processing failed",
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if operation:
            details["operation"] = operation
        super().__init__(message=message, details=details)


class AudioEncodingError(AudioProcessingError):
    """Raised when audio encoding fails (500)."""

    error_code = "TTS_E511"

    def __init__(self, format: str, reason: str | None = None) -> None:
        details: dict[str, Any] = {"format": format}
        if reason:
            details["reason"] = reason
        super().__init__(
            message=f"Failed to encode audio to {format}",
            operation="encoding",
            details=details,
        )


class AudioDecodingError(AudioProcessingError):
    """Raised when audio decoding fails (500)."""

    error_code = "TTS_E512"

    def __init__(self, format: str | None = None, reason: str | None = None) -> None:
        details: dict[str, Any] = {}
        if format:
            details["format"] = format
        if reason:
            details["reason"] = reason
        super().__init__(
            message="Failed to decode audio",
            operation="decoding",
            details=details,
        )


# =============================================================================
# 503 Service Unavailable - Resource Errors
# =============================================================================


class ResourceError(TTSBaseException):
    """Raised when a required resource is unavailable (503)."""

    error_code = "TTS_E600"
    http_status = 503

    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,
    ) -> None:
        details = details or {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message=message, details=details)
        self.retry_after = retry_after


class GPUUnavailableError(ResourceError):
    """Raised when GPU is unavailable (503)."""

    error_code = "TTS_E601"

    def __init__(self, reason: str | None = None) -> None:
        details: dict[str, Any] = {}
        if reason:
            details["reason"] = reason
        super().__init__(
            message="GPU is unavailable for inference",
            details=details,
        )


class OutOfMemoryError(ResourceError):
    """Raised when system runs out of memory (503)."""

    error_code = "TTS_E602"

    def __init__(self, memory_type: str = "GPU") -> None:
        super().__init__(
            message=f"Out of {memory_type} memory",
            details={"memory_type": memory_type},
            retry_after=30,
        )


class ModelNotLoadedError(ResourceError):
    """Raised when the TTS model is not loaded (503)."""

    error_code = "TTS_E603"

    def __init__(self) -> None:
        super().__init__(
            message="TTS model is not loaded. Service is starting up.",
            retry_after=10,
        )


class ServiceOverloadedError(ResourceError):
    """Raised when service is overloaded (503)."""

    error_code = "TTS_E604"

    def __init__(self, queue_size: int | None = None) -> None:
        details: dict[str, Any] = {}
        if queue_size is not None:
            details["queue_size"] = queue_size
        super().__init__(
            message="Service is currently overloaded. Please try again later.",
            details=details,
            retry_after=60,
        )


# Legacy aliases for backward compatibility
APIException = TTSBaseException
SynthesisError = InferenceError
RateLimitExceededError = RateLimitError
