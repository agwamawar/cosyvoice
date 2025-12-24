"""Global error handling middleware and exception handlers.

Provides:
- Handler for TTSBaseException returning structured JSON
- Handler for RequestValidationError (Pydantic) returning friendly messages
- Handler for generic Exception logging full traceback, returning safe 500
- Response format: {"error": {"code": "...", "message": "...", "details": {...}, "request_id": "..."}}
"""

import traceback
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.exceptions import (
    RateLimitError,
    ResourceError,
    TTSBaseException,
)
from src.config import get_settings
from src.utils.logging import get_log_context, get_logger

logger = get_logger("api")


def _get_request_id() -> str | None:
    """Get request ID from current context."""
    context = get_log_context()
    return context.get("request_id")


def _build_error_response(
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build standardized error response structure."""
    error_body: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if details:
        error_body["details"] = details
    if request_id:
        error_body["request_id"] = request_id
    return {"error": error_body}


async def tts_exception_handler(
    request: Request,
    exc: TTSBaseException,
) -> JSONResponse:
    """Handle TTSBaseException and subclasses."""
    request_id = _get_request_id()

    # Log the exception
    logger.warning(
        "TTS exception",
        error_code=exc.error_code,
        error_message=exc.message,
        http_status=exc.http_status,
        details=exc.details,
        path=request.url.path,
    )

    # Build response headers
    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    # Add Retry-After header for rate limits and resource errors
    if isinstance(exc, RateLimitError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)
    elif isinstance(exc, ResourceError) and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=exc.http_status,
        content=_build_error_response(
            code=exc.error_code,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
        ),
        headers=headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors with friendly messages."""
    request_id = _get_request_id()

    # Parse validation errors into friendly format
    errors: list[dict[str, Any]] = []
    for error in exc.errors():
        loc = error.get("loc", ())
        # Build field path (skip 'body' prefix)
        field_parts = [str(part) for part in loc if part != "body"]
        field = ".".join(field_parts) if field_parts else "unknown"

        errors.append({
            "field": field,
            "message": error.get("msg", "Invalid value"),
            "type": error.get("type", "value_error"),
        })

    # Log validation failure
    logger.warning(
        "Validation error",
        path=request.url.path,
        error_count=len(errors),
        fields=[e["field"] for e in errors],
    )

    # Build user-friendly message
    if len(errors) == 1:
        message = f"Validation error: {errors[0]['message']} (field: {errors[0]['field']})"
    else:
        message = f"Validation failed with {len(errors)} errors"

    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=400,
        content=_build_error_response(
            code="TTS_E100",
            message=message,
            details={"validation_errors": errors},
            request_id=request_id,
        ),
        headers=headers,
    )


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle Starlette HTTP exceptions."""
    request_id = _get_request_id()

    # Map common HTTP status codes to error codes
    code_map = {
        400: "TTS_E100",
        401: "TTS_E200",
        403: "TTS_E200",
        404: "TTS_E300",
        405: "TTS_E100",
        429: "TTS_E400",
        500: "TTS_E000",
        503: "TTS_E600",
    }

    error_code = code_map.get(exc.status_code, "TTS_E000")

    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
    )

    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    return JSONResponse(
        status_code=exc.status_code,
        content=_build_error_response(
            code=error_code,
            message=str(exc.detail) if exc.detail else "An error occurred",
            request_id=request_id,
        ),
        headers=headers,
    )


async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions safely."""
    request_id = _get_request_id()
    settings = get_settings()

    # Log full traceback
    logger.exception(
        "Unhandled exception",
        error_type=type(exc).__name__,
        error=str(exc),
        path=request.url.path,
        method=request.method,
    )

    headers: dict[str, str] = {}
    if request_id:
        headers["X-Request-ID"] = request_id

    # In debug mode, include more details
    if settings.app.debug:
        return JSONResponse(
            status_code=500,
            content=_build_error_response(
                code="TTS_E000",
                message=str(exc),
                details={
                    "type": type(exc).__name__,
                    "traceback": traceback.format_exc().split("\n"),
                },
                request_id=request_id,
            ),
            headers=headers,
        )

    # Production: safe generic message
    return JSONResponse(
        status_code=500,
        content=_build_error_response(
            code="TTS_E000",
            message="An unexpected error occurred. Please try again later.",
            request_id=request_id,
        ),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(TTSBaseException, tts_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
