"""Application settings using pydantic-settings."""

from enum import Enum
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment."""

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class AudioFormat(str, Enum):
    """Supported audio output formats."""

    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"


class AppSettings(BaseSettings):
    """Application-level settings."""

    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    name: str = Field(default="TTS Backend", alias="APP_NAME")
    version: str = Field(default="0.1.0", alias="APP_VERSION")
    environment: Environment = Field(default=Environment.DEV, alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PROD


class ServerSettings(BaseSettings):
    """Server configuration settings."""

    model_config = SettingsConfigDict(env_prefix="SERVER_", env_file=".env", extra="ignore")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    reload: bool = Field(default=False)


class ModelSettings(BaseSettings):
    """ML Model configuration settings."""

    model_config = SettingsConfigDict(env_prefix="MODEL_", env_file=".env", extra="ignore")

    path: str = Field(default="./models/cosyvoice", alias="MODEL_PATH")
    device: Literal["cuda", "cpu", "auto"] = Field(default="auto")
    dtype: Literal["float32", "float16", "bfloat16"] = Field(default="float32")
    voices_directory: str = Field(default="./voices")
    use_mock: bool = Field(default=True)

    def get_torch_device(self) -> str:
        """Get the actual torch device to use."""
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def get_torch_dtype(self) -> "Any":
        """Get the torch dtype object."""
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map[self.dtype]


class AudioSettings(BaseSettings):
    """Audio processing settings."""

    model_config = SettingsConfigDict(env_prefix="AUDIO_", env_file=".env", extra="ignore")

    default_sample_rate: int = Field(default=22050, ge=8000, le=48000)
    default_format: AudioFormat = Field(default=AudioFormat.WAV)
    max_input_duration: int = Field(default=300, ge=1, description="Max input duration in seconds")
    max_output_duration: int = Field(
        default=600, ge=1, description="Max output duration in seconds"
    )
    max_file_size_mb: int = Field(default=50, ge=1, le=500)

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


class APISettings(BaseSettings):
    """API security and rate limiting settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    keys: str = Field(default="", alias="API_KEYS")
    rate_limit_requests: int = Field(default=100, ge=1, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, ge=1, alias="RATE_LIMIT_WINDOW")

    @property
    def api_keys(self) -> list[str]:
        """Get parsed API keys as list."""
        if not self.keys:
            return []
        return [k.strip() for k in self.keys.split(",") if k.strip()]

    @property
    def auth_enabled(self) -> bool:
        """Check if API authentication is enabled."""
        return len(self.api_keys) > 0


class Settings(BaseSettings):
    """Combined application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app: AppSettings = Field(default_factory=AppSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    api: APISettings = Field(default_factory=APISettings)

    # Logging settings
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "pretty"] = Field(default="pretty", alias="LOG_FORMAT")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
