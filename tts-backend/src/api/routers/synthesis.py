"""Speech synthesis endpoints.

Prefix: /v1/tts
Requires API key authentication.
"""

import base64
import time
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from src.api.dependencies import APIKeyDep, RequestIdDep, TTSServiceDep
from src.api.exceptions import VoiceNotFoundError
from src.api.models.requests import SynthesizeRequest
from src.api.models.responses import SynthesizeResponse
from src.audio.formats import AudioFormat

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
    # Verify voice exists
    voice = await tts_service.get_voice(request.voice_id)
    if voice is None:
        raise VoiceNotFoundError(request.voice_id)

    # Build synthesis options
    from src.core.interfaces import SynthesisOptions

    options = SynthesisOptions(
        voice_id=request.voice_id,
        speed=request.speed,
        language=request.language,
        output_format=AudioFormat(request.output_format),
        sample_rate=request.sample_rate,
    )

    # Perform synthesis
    audio_result = await tts_service.synthesize(request.text, options)

    # Get content type
    content_type = CONTENT_TYPE_MAP.get(request.output_format, "audio/wav")

    return StreamingResponse(
        iter([audio_result.audio_data]),
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=speech.{request.output_format}",
            "Content-Length": str(len(audio_result.audio_data)),
            "X-Duration-Seconds": str(audio_result.duration_seconds),
            "X-Voice-ID": request.voice_id,
            "X-Sample-Rate": str(audio_result.sample_rate),
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

    # Verify voice exists
    voice = await tts_service.get_voice(request.voice_id)
    if voice is None:
        raise VoiceNotFoundError(request.voice_id)

    # Build synthesis options
    from src.core.interfaces import SynthesisOptions

    options = SynthesisOptions(
        voice_id=request.voice_id,
        speed=request.speed,
        language=request.language,
        output_format=AudioFormat(request.output_format),
        sample_rate=request.sample_rate,
    )

    # Perform synthesis
    audio_result = await tts_service.synthesize(request.text, options)

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    # Encode audio as base64
    audio_base64 = base64.b64encode(audio_result.audio_data).decode("utf-8")

    return SynthesizeResponse(
        audio_base64=audio_base64,
        format=request.output_format,
        sample_rate=audio_result.sample_rate,
        duration_seconds=audio_result.duration_seconds,
        request_id=request_id,
        voice_id=request.voice_id,
        text_length=len(request.text),
        processing_time_ms=round(processing_time_ms, 2),
    )
