"""Speech synthesis endpoints.

Prefix: /v1/tts
Requires API key authentication.
"""

import base64
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import APIKeyDep, TTSServiceDep
from src.api.exceptions import VoiceNotFoundError
from src.api.models.requests import SynthesizeRequest
from src.api.models.responses import SynthesizeResponse
from src.audio.formats import AudioFormat
from src.core.models import SynthesisRequest as CoreSynthesisRequest

router = APIRouter(prefix="/v1/tts", tags=["Synthesis"])

# Content type mapping
CONTENT_TYPE_MAP = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


@router.post("/synthesize")
async def synthesize(
    request: SynthesizeRequest,
    tts_service: TTSServiceDep,
    _api_key: APIKeyDep,
) -> StreamingResponse:
    """
    Synthesize speech from text and return as audio file.

    Returns audio with appropriate Content-Type header.
    Use Content-Disposition for filename suggestion.
    """
    # Build core synthesis request
    core_request = CoreSynthesisRequest(
        text=request.text,
        voice_id=request.voice_id,
        speed=request.speed,
        language=request.language,
        output_format=request.output_format,
        sample_rate=request.sample_rate,
    )

    # Perform synthesis
    synthesis_response = await tts_service.synthesize(core_request)

    # Decode base64 audio to bytes
    audio_data = base64.b64decode(synthesis_response.audio_base64)

    # Get content type
    content_type = CONTENT_TYPE_MAP.get(request.output_format, "audio/wav")

    return StreamingResponse(
        iter([audio_data]),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{request.output_format}",
            "Content-Length": str(len(audio_data)),
            "X-Duration-Seconds": str(synthesis_response.duration_seconds),
            "X-Voice-ID": request.voice_id,
            "X-Sample-Rate": str(synthesis_response.sample_rate),
        },
    )


@router.post("/synthesize/json", response_model=SynthesizeResponse)
async def synthesize_json(
    request: SynthesizeRequest,
    http_request: Request,
    tts_service: TTSServiceDep,
    _api_key: APIKeyDep,
) -> SynthesizeResponse:
    """
    Synthesize speech from text and return as JSON with base64 audio.

    Same as /synthesize but returns JSON response with:
    - audio_base64: Base64-encoded audio data
    - Metadata about the synthesis
    """
    start_time = time.perf_counter()
    request_id = getattr(http_request.state, "request_id", str(uuid.uuid4()))

    # Build core synthesis request
    core_request = CoreSynthesisRequest(
        text=request.text,
        voice_id=request.voice_id,
        speed=request.speed,
        language=request.language,
        output_format=request.output_format,
        sample_rate=request.sample_rate,
    )

    # Perform synthesis
    synthesis_response = await tts_service.synthesize(core_request)

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    return SynthesizeResponse(
        audio_base64=synthesis_response.audio_base64,
        format=request.output_format,
        sample_rate=synthesis_response.sample_rate,
        duration_seconds=synthesis_response.duration_seconds,
        request_id=request_id,
        voice_id=request.voice_id,
        text_length=len(request.text),
        processing_time_ms=round(processing_time_ms, 2),
    )
