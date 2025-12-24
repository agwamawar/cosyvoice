"""Integration tests for health endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    async def test_health_endpoint_returns_ok(self, client: AsyncClient):
        """Test basic health check returns 200 OK."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data

    async def test_ready_endpoint_shows_status(self, client: AsyncClient):
        """Test readiness probe shows engine status."""
        response = await client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "ready" in data
        assert isinstance(data["ready"], bool)
        assert "checks" in data
        assert isinstance(data["checks"], dict)

    async def test_live_endpoint(self, client: AsyncClient):
        """Test liveness probe returns alive status."""
        response = await client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    async def test_health_no_auth_required(self, client: AsyncClient):
        """Test health endpoints don't require authentication."""
        # All health endpoints should work without X-API-Key
        endpoints = ["/health", "/health/ready", "/health/live"]

        for endpoint in endpoints:
            response = await client.get(endpoint)
            assert response.status_code == 200, f"{endpoint} should not require auth"
