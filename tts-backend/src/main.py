"""Main entry point for the TTS Backend service.

Usage:
    python -m src.main
    uvicorn src.main:app --reload
"""

from src.api.app import create_app
from src.config import get_settings

# Create FastAPI application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "src.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
    )
