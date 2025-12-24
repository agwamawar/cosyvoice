"""Request context middleware for tracking requests.

Provides:
- Extracts or generates X-Request-ID header
- Sets up logging context for the request (request_id, user_id, etc.)
- Adds request_id to response headers
- Cleans up context after request
"""

import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from src.utils.logging import LogContext, clear_log_context, get_logger, set_log_context

logger = get_logger("api")

# Context variable for request-scoped data
_request_context: ContextVar["RequestContext | None"] = ContextVar(
    "request_context", default=None
)


@dataclass
class RequestContext:
    """Context data for the current request."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: float = field(default_factory=time.perf_counter)
    start_datetime: datetime = field(default_factory=datetime.utcnow)
    method: str = ""
    path: str = ""
    user_agent: str | None = None
    client_ip: str | None = None
    user_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000

    def set_user(self, user_id: str) -> None:
        """Set the user ID for this request."""
        self.user_id = user_id
        set_log_context(user_id=user_id)

    def add_extra(self, **kwargs: Any) -> None:
        """Add extra context data."""
        self.extra.update(kwargs)
        set_log_context(**kwargs)


def get_request_context() -> RequestContext | None:
    """Get the current request context."""
    return _request_context.get()


def get_request_id() -> str | None:
    """Get the current request ID."""
    ctx = get_request_context()
    return ctx.request_id if ctx else None


def set_request_user(user_id: str) -> None:
    """Set the user ID for the current request."""
    ctx = get_request_context()
    if ctx:
        ctx.set_user(user_id)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware to set up request context for each request.

    This middleware:
    1. Extracts or generates X-Request-ID from headers
    2. Sets up logging context with request_id for all log messages
    3. Tracks request metadata (client IP, user agent, etc.)
    4. Adds X-Request-ID to response headers
    5. Logs request completion with timing
    6. Cleans up context after request
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Set up request context and process the request."""
        # Extract or generate request ID
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())

        # Create request context
        context = RequestContext(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            user_agent=request.headers.get("User-Agent"),
            client_ip=self._get_client_ip(request),
        )

        # Set context variable
        context_token = _request_context.set(context)

        # Set up logging context - this makes request_id available to all loggers
        with LogContext(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        ):
            # Log request start (debug level to avoid noise)
            logger.debug(
                "Request started",
                client_ip=context.client_ip,
                user_agent=context.user_agent,
            )

            try:
                # Process the request
                response = await call_next(request)

                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id

                # Add timing header
                response.headers["X-Request-Duration-Ms"] = f"{context.elapsed_ms:.2f}"

                # Log request completion
                logger.info(
                    "Request completed",
                    status_code=response.status_code,
                    duration_ms=round(context.elapsed_ms, 2),
                )

                return response

            except Exception as exc:
                # Log exception (will be handled by error handler)
                logger.error(
                    "Request failed with exception",
                    error_type=type(exc).__name__,
                    duration_ms=round(context.elapsed_ms, 2),
                )
                raise

            finally:
                # Reset request context
                _request_context.reset(context_token)

    def _get_client_ip(self, request: Request) -> str | None:
        """Extract client IP from request, handling proxies."""
        # Check for forwarded headers (reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client connection
        if request.client:
            return request.client.host

        return None


def create_request_context(
    request_id: str | None = None,
    **kwargs: Any,
) -> RequestContext:
    """
    Create a request context manually (useful for background tasks).

    Args:
        request_id: Optional request ID (generated if not provided)
        **kwargs: Additional context fields

    Returns:
        RequestContext instance
    """
    context = RequestContext(
        request_id=request_id or str(uuid.uuid4()),
        **kwargs,
    )
    _request_context.set(context)
    set_log_context(request_id=context.request_id, **kwargs)
    return context


def clear_request_context() -> None:
    """Clear the current request context (useful for cleanup in background tasks)."""
    _request_context.set(None)
    clear_log_context()
