# API Documentation

## Base URL

```
http://localhost:8000
```

Production (RunPod):
```
https://{POD_ID}-8000.proxy.runpod.net
```

## Authentication

All endpoints except health checks require an API key in the `X-API-Key` header.

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/v1/voices
```

## Health Endpoints

Health endpoints do not require authentication.

### GET /health

Service health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "environment": "development",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### GET /health/ready

Readiness probe - checks if the model is loaded and ready.

**Response:**
```json
{
  "ready": true,
  "checks": {
    "model_loaded": true,
    "voices_available": true
  }
}
```

### GET /health/live

Liveness probe - simple alive check.

**Response:**
```json
{
  "status": "alive"
}
```

---

## TTS Endpoints

### POST /v1/tts/synthesize

Synthesize speech from text. Returns audio file directly.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "Hello, world!",
    "voice_id": "default",
    "output_format": "wav",
    "sample_rate": 22050,
    "speed": 1.0
  }' \
  --output speech.wav
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | - | Text to synthesize (1-5000 chars) |
| `voice_id` | string | No | `"default"` | Voice identifier |
| `output_format` | string | No | `"wav"` | Output format: wav, mp3, ogg, flac |
| `sample_rate` | integer | No | `22050` | Sample rate: 8000-48000 |
| `speed` | float | No | `1.0` | Speech speed: 0.5-2.0 |
| `language` | string | No | `"en"` | Language code |

**Response:**
- Content-Type: `audio/wav` (or appropriate for format)
- Content-Disposition: `attachment; filename="speech.wav"`
- Body: Raw audio bytes

### POST /v1/tts/synthesize/json

Synthesize speech and return as base64 JSON.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/tts/synthesize/json \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "Hello, world!",
    "voice_id": "default"
  }'
```

**Response:**
```json
{
  "audio_base64": "UklGRi4AAABXQVZFZm10IBAAAAABAAEA...",
  "format": "wav",
  "sample_rate": 22050,
  "duration_seconds": 1.5,
  "request_id": "req_abc123"
}
```

### POST /v1/tts/synthesize/stream

Streaming synthesis - returns audio chunks as they're generated.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/tts/synthesize/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "This is a longer text that will be streamed...",
    "voice_id": "default"
  }' \
  --output stream.wav
```

**Response:**
- Content-Type: `audio/wav`
- Transfer-Encoding: `chunked`
- Body: Streamed audio chunks

---

## Voice Endpoints

### GET /v1/voices

List all available voices.

**Request:**
```bash
curl http://localhost:8000/v1/voices \
  -H "X-API-Key: your-api-key"
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `language` | string | Filter by language code |
| `gender` | string | Filter by gender: male, female, neutral |

**Response:**
```json
{
  "voices": [
    {
      "voice_id": "default",
      "name": "Default",
      "language": "en",
      "gender": "neutral",
      "is_cloned": false,
      "description": "Default English voice"
    },
    {
      "voice_id": "en-female-1",
      "name": "English Female 1",
      "language": "en",
      "gender": "female",
      "is_cloned": false,
      "description": "Natural English female voice"
    }
  ],
  "total": 2
}
```

### GET /v1/voices/{voice_id}

Get details for a specific voice.

**Request:**
```bash
curl http://localhost:8000/v1/voices/default \
  -H "X-API-Key: your-api-key"
```

**Response:**
```json
{
  "voice_id": "default",
  "name": "Default",
  "language": "en",
  "gender": "neutral",
  "is_cloned": false,
  "description": "Default English voice"
}
```

**Error (404):**
```json
{
  "error": {
    "code": "VOICE_NOT_FOUND",
    "message": "Voice 'unknown-voice' not found"
  }
}
```

### POST /v1/voices/clone

Clone a voice from an audio sample.

**Request:**
```bash
curl -X POST http://localhost:8000/v1/voices/clone \
  -H "X-API-Key: your-api-key" \
  -F "audio=@sample.wav" \
  -F "voice_name=My Custom Voice" \
  -F "description=Cloned from my recording"
```

**Form Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | Yes | Audio file (WAV, 3-30 seconds) |
| `voice_name` | string | Yes | Name for the cloned voice |
| `description` | string | No | Voice description |

**Response (201):**
```json
{
  "voice_id": "cloned_abc123",
  "name": "My Custom Voice",
  "language": "en",
  "gender": null,
  "is_cloned": true,
  "description": "Cloned from my recording"
}
```

### DELETE /v1/voices/{voice_id}

Delete a cloned voice. Built-in voices cannot be deleted.

**Request:**
```bash
curl -X DELETE http://localhost:8000/v1/voices/cloned_abc123 \
  -H "X-API-Key: your-api-key"
```

**Response (204):**
No content.

**Error (400):**
```json
{
  "error": {
    "code": "CANNOT_DELETE_BUILTIN",
    "message": "Cannot delete built-in voice 'default'"
  }
}
```

---

## Error Response Format

All errors follow a consistent format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "additional context"
    }
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `VALIDATION_ERROR` | 400 | Invalid request parameters |
| `CANNOT_DELETE_BUILTIN` | 400 | Attempted to delete built-in voice |
| `MISSING_API_KEY` | 401 | No X-API-Key header provided |
| `INVALID_API_KEY` | 401 | Invalid API key |
| `VOICE_NOT_FOUND` | 404 | Requested voice doesn't exist |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `SYNTHESIS_ERROR` | 500 | TTS synthesis failed |
| `MODEL_NOT_LOADED` | 503 | Model not ready |

---

## Rate Limiting

API requests are rate-limited per API key.

**Default Limits:**
- 100 requests per 60 seconds

**Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

**Rate Limit Exceeded (429):**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 45 seconds.",
    "details": {
      "retry_after": 45
    }
  }
}
```

---

## Examples

### Python

```python
import httpx
import base64

API_KEY = "your-api-key"
BASE_URL = "http://localhost:8000"

# Synthesize and save audio
response = httpx.post(
    f"{BASE_URL}/v1/tts/synthesize",
    headers={"X-API-Key": API_KEY},
    json={"text": "Hello, world!", "voice_id": "default"},
)
with open("output.wav", "wb") as f:
    f.write(response.content)

# Get base64 audio
response = httpx.post(
    f"{BASE_URL}/v1/tts/synthesize/json",
    headers={"X-API-Key": API_KEY},
    json={"text": "Hello, world!"},
)
data = response.json()
audio_bytes = base64.b64decode(data["audio_base64"])
```

### JavaScript

```javascript
const API_KEY = "your-api-key";
const BASE_URL = "http://localhost:8000";

// Synthesize speech
const response = await fetch(`${BASE_URL}/v1/tts/synthesize`, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY,
  },
  body: JSON.stringify({
    text: "Hello, world!",
    voice_id: "default",
  }),
});

const audioBlob = await response.blob();
const audioUrl = URL.createObjectURL(audioBlob);
const audio = new Audio(audioUrl);
audio.play();
```

### cURL

```bash
# List voices
curl -H "X-API-Key: your-api-key" http://localhost:8000/v1/voices

# Synthesize to file
curl -X POST http://localhost:8000/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"text": "Hello!", "voice_id": "default"}' \
  -o output.wav

# Clone voice
curl -X POST http://localhost:8000/v1/voices/clone \
  -H "X-API-Key: your-api-key" \
  -F "audio=@sample.wav" \
  -F "voice_name=My Voice"
```
