"""Integration tests for synthesis endpoints."""

import base64

import pytest
from httpx import AsyncClient


@pytest.mark.integration
class TestSynthesisEndpoints:
    """Tests for speech synthesis endpoints."""

    @pytest.mark.skip(reason="Requires torchaudio backend (sox/soundfile)")
    async def test_synthesize_returns_audio_file(
        self,
        client: AsyncClient,
        synthesis_request_data: dict,
        auth_headers: dict,
    ):
        """Test synthesis returns audio file with proper headers."""
        response = await client.post(
            "/v1/tts/synthesize",
            json=synthesis_request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert "audio/" in response.headers["content-type"]
        assert "attachment" in response.headers.get("content-disposition", "")
        assert len(response.content) > 0

    async def test_synthesize_without_auth_returns_401(
        self,
        client: AsyncClient,
        synthesis_request_data: dict,
        no_auth_headers: dict,
    ):
        """Test synthesis without API key returns 401."""
        response = await client.post(
            "/v1/tts/synthesize",
            json=synthesis_request_data,
            headers=no_auth_headers,
        )

        assert response.status_code == 401

    async def test_synthesize_empty_text_returns_400(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        """Test synthesis with empty text returns 400 validation error."""
        response = await client.post(
            "/v1/tts/synthesize",
            json={"text": ""},
            headers=auth_headers,
        )

        # API returns 400 for validation errors (not 422)
        assert response.status_code == 400

    @pytest.mark.skip(reason="Requires torchaudio backend (sox/soundfile)")
    async def test_synthesize_json_returns_base64(
        self,
        client: AsyncClient,
        synthesis_request_data: dict,
        auth_headers: dict,
    ):
        """Test JSON synthesis endpoint returns base64 audio."""
        response = await client.post(
            "/v1/tts/synthesize/json",
            json=synthesis_request_data,
            headers=auth_headers,
        )

        assert response.status_code == 200
        data = response.json()

        assert "audio_base64" in data
        assert "format" in data
        assert "sample_rate" in data
        assert "duration_seconds" in data
        assert "request_id" in data

        # Verify audio_base64 is valid base64
        audio_bytes = base64.b64decode(data["audio_base64"])
        assert len(audio_bytes) > 0

    async def test_synthesize_invalid_voice_returns_404(
        self,
        client: AsyncClient,
        auth_headers: dict,
    ):
        """Test synthesis with invalid voice ID returns 404."""
        response = await client.post(
            "/v1/tts/synthesize",
            json={
                "text": "Hello, world!",
                "voice_id": "nonexistent-voice-id",
            },
            headers=auth_headers,
        )

        assert response.status_code == 404
