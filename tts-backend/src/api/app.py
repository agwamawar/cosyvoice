"""FastAPI application factory.

Provides:
- create_app(): Application factory function
- Lifespan context manager for startup/shutdown
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.error_handler import register_exception_handlers
from src.api.middleware.request_context import RequestContextMiddleware
from src.api.routers import health, synthesis, voices
from src.config import Settings, get_settings
from src.core.tts_service import TTSService
from src.models import register_engines
from src.utils.logging import get_logger, setup_logging

logger = get_logger("system")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan handler for startup/shutdown.

    On startup:
    - Initialize logging
    - Register engines
    - Initialize TTS service
    - Log startup complete

    On shutdown:
    - Cleanup TTS service
    - Unload engines
    """
    settings = get_settings()

    # Setup logging on startup
    setup_logging(
        level=settings.log_level,
        format=settings.log_format,
        service_name=settings.app.name,
    )

    logger.info(
        "Starting TTS Backend",
        extra={
            "version": settings.app.version,
            "environment": settings.app.environment.value,
            "debug": settings.app.debug,
        },
    )

    # Register engines
    try:
        register_engines()
        logger.info("Engines registered successfully")
    except Exception as e:
        logger.warning(f"Engine registration failed: {e}")

    # Initialize TTS service
    tts_service = TTSService(settings)
    await tts_service.initialize()
    app.state.tts_service = tts_service

    logger.info("TTS Backend ready to accept requests")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down TTS Backend")
    await tts_service.shutdown()
    logger.info("TTS Backend shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional settings override (uses get_settings() if None)

    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="TTS Backend",
        version=settings.app.version,
        description="Text-to-Speech API powered by CosyVoice 3",
        docs_url="/docs" if settings.app.debug else None,
        redoc_url="/redoc" if settings.app.debug else None,
        openapi_url="/openapi.json" if settings.app.debug else None,
        lifespan=lifespan,
    )

    # Register exception handlers (for structured error responses)
    register_exception_handlers(app)

    # Add middleware (order matters - first added is outermost)
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins if hasattr(settings.api, 'cors_origins') else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request context middleware (innermost - wraps entire request)
    app.add_middleware(RequestContextMiddleware)

    # Include routers
    # Health router - no prefix (endpoints define their own prefix)
    app.include_router(health.router)

    # Synthesis router - prefix defined in router
    app.include_router(synthesis.router)

    # Voices router - prefix defined in router
    app.include_router(voices.router)

    return app
