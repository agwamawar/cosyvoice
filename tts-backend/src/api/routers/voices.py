"""Voice management endpoints.

Prefix: /v1/voices
Requires API key authentication.
"""

from fastapi import APIRouter, File, Form, UploadFile

from src.api.dependencies import APIKeyDep, TTSServiceDep
from src.api.exceptions import ValidationError, VoiceNotFoundError
from src.api.models.responses import VoiceListResponse, VoiceResponse

router = APIRouter(prefix="/v1/voices", tags=["Voices"])


@router.get("", response_model=VoiceListResponse)
@router.get("/", response_model=VoiceListResponse, include_in_schema=False)
async def list_voices(
    tts_service: TTSServiceDep,
    _api_key: APIKeyDep,
    language: str | None = None,
) -> VoiceListResponse:
    """
    List all available voices.

    Optionally filter by language code.
    """
    voices = await tts_service.list_voices(language=language)

    return VoiceListResponse(
        voices=[
            VoiceResponse(
                voice_id=v.voice_id,
                name=v.name,
                language=v.language,
                gender=v.gender,
                is_cloned=v.is_cloned,
                description=v.description,
            )
            for v in voices
        ],
        total=len(voices),
    )


@router.get("/{voice_id}", response_model=VoiceResponse)
async def get_voice(
    voice_id: str,
    tts_service: TTSServiceDep,
    _api_key: APIKeyDep,
) -> VoiceResponse:
    """
    Get details for a specific voice.

    Returns 404 if voice not found.
    """
    voice = await tts_service.get_voice(voice_id)

    if voice is None:
        raise VoiceNotFoundError(voice_id)

    return VoiceResponse(
        voice_id=voice.voice_id,
        name=voice.name,
        language=voice.language,
        gender=voice.gender,
        is_cloned=voice.is_cloned,
        description=voice.description,
    )


@router.post("/clone", response_model=VoiceResponse, status_code=201)
async def clone_voice(
    tts_service: TTSServiceDep,
    _api_key: APIKeyDep,
    audio: UploadFile = File(..., description="Reference audio file for voice cloning"),
    voice_name: str = Form(..., min_length=1, max_length=100),
    description: str | None = Form(default=None, max_length=500),
) -> VoiceResponse:
    """
    Clone a voice from an audio sample.

    Upload an audio file (WAV, MP3, etc.) and provide a name for the voice.
    The audio should be 3-30 seconds of clear speech.
    """
    # Validate file type
    content_type = audio.content_type or ""
    if not content_type.startswith("audio/"):
        raise ValidationError(
            message="Invalid file type. Please upload an audio file.",
            details={"content_type": content_type},
        )

    # Read audio content
    audio_content = await audio.read()

    # Validate file size (max 50MB)
    if len(audio_content) > 50 * 1024 * 1024:
        raise ValidationError(
            message="File too large. Maximum size is 50MB.",
            details={"size_bytes": len(audio_content)},
        )

    # Clone voice
    voice = await tts_service.clone_voice(
        voice_name=voice_name,
        audio_data=audio_content,
        description=description,
    )

    return VoiceResponse(
        voice_id=voice.voice_id,
        name=voice.name,
        language=voice.language,
        gender=voice.gender,
        is_cloned=voice.is_cloned,
        description=voice.description,
    )


@router.delete("/{voice_id}")
async def delete_voice(
    voice_id: str,
    tts_service: TTSServiceDep,
    _api_key: APIKeyDep,
) -> dict[str, bool]:
    """
    Delete a custom/cloned voice.

    Returns 404 if voice not found.
    Returns 400 if trying to delete a built-in voice.
    """
    voice = await tts_service.get_voice(voice_id)

    if voice is None:
        raise VoiceNotFoundError(voice_id)

    if not voice.is_cloned:
        raise ValidationError(
            message="Cannot delete built-in voices",
            details={"voice_id": voice_id},
        )

    await tts_service.delete_voice(voice_id)

    return {"deleted": True}
