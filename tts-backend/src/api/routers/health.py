"""Health check endpoints.

Prefix: /health
No authentication required.
"""

from datetime import datetime

from fastapi import APIRouter, Request

from src.api.models.responses import HealthResponse, ReadinessResponse
from src.config import get_settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
@router.get("/", response_model=HealthResponse, include_in_schema=False)
async def health_check() -> HealthResponse:
    """
    Basic health check - returns service status.

    No authentication required. Use for basic "is the service alive" checks.
    """
    settings = get_settings()

    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.app.version,
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check(request: Request) -> ReadinessResponse:
    """
    Readiness probe - checks if the service can accept traffic.

    Verifies:
    - TTS engine is loaded
    - Service is ready to handle requests

    Use for Kubernetes readiness probes.
    """
    checks = {
        "config_valid": True,
    }

    # Check if TTS service is available
    try:
        tts_service = request.app.state.tts_service
        engine_loaded = tts_service.is_ready()
        checks["engine_loaded"] = engine_loaded
    except Exception:
        checks["engine_loaded"] = False

    return ReadinessResponse(
        ready=all(checks.values()),
        checks=checks,
    )


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """
    Liveness probe - basic check that the service is running.

    Always returns {"status": "alive"} if the service is responding.
    Use for Kubernetes liveness probes.
    """
    return {"status": "alive"}
