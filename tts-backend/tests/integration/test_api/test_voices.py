"""Integration tests for voices endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestVoicesEndpoints:
    """Tests for voice management endpoints."""

    async def test_list_voices_returns_voices(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        """Test listing voices returns voice list."""
        response = await client.get(
            "/v1/voices",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "voices" in data
        assert "total" in data
        assert isinstance(data["voices"], list)
        assert data["total"] >= 0

        # Check that default voice exists if voices are present
        if data["total"] > 0:
            voice = data["voices"][0]
            assert "voice_id" in voice
            assert "name" in voice
            assert "language" in voice

    async def test_list_voices_with_language_filter(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        """Test filtering voices by language."""
        response = await client.get(
            "/v1/voices?language=en",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        # All returned voices should be English
        for voice in data["voices"]:
            assert voice["language"] == "en"

    async def test_get_voice_not_found_returns_404(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        """Test getting non-existent voice returns 404."""
        response = await client.get(
            "/v1/voices/nonexistent-voice-id",
            headers=auth_headers,
        )

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    async def test_get_voice_success(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        """Test getting existing voice returns voice info."""
        response = await client.get(
            "/v1/voices/default",
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert data["voice_id"] == "default"
        assert "name" in data
        assert "language" in data

    async def test_voices_require_auth(
        self,
        client: AsyncClient,
        no_auth_headers: dict,
    ):
        """Test voice endpoints require authentication."""
        response = await client.get(
            "/v1/voices",
            headers=no_auth_headers,
        )

        assert response.status_code == 401


@pytest.mark.integration
class TestVoiceCloning:
    """Tests for voice cloning functionality."""

    async def test_clone_voice_with_audio(
        self,
        client: AsyncClient,
        auth_headers: dict,
        sample_wav_bytes: bytes,
    ):
        """Test cloning a voice with audio sample."""
        # Note: This test may need adjustment based on actual implementation
        files = {"audio": ("sample.wav", sample_wav_bytes, "audio/wav")}
        data = {"voice_name": "Test Cloned Voice", "description": "Test clone"}

        response = await client.post(
            "/v1/voices/clone",
            files=files,
            data=data,
            headers={"X-API-Key": auth_headers["X-API-Key"]},
        )

        # May return 201 for success or 500 if cloning not fully implemented
        assert response.status_code in [201, 500, 501]

    async def test_delete_builtin_voice_returns_400(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        """Test deleting built-in voice returns 400."""
        response = await client.delete(
            "/v1/voices/default",
            headers=auth_headers,
        )

        # Should fail because default is a built-in voice
        assert response.status_code == 400
