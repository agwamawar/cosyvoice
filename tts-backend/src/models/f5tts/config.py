"""Configuration for F5-TTS model."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class F5TTSConfig(BaseSettings):
    """Configuration for F5-TTS model."""

    # Model settings
    model_name: str = Field(
        default="F5-TTS",
        description="Model name (F5-TTS or E2-TTS)",
    )
    model_path: str = Field(
        default="/workspace/models/F5-TTS",
        description="Path to model directory (optional, uses default if not set)",
    )

    # Device settings
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Device to run inference on (cuda, cpu, or auto)",
    )
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",
        description="Model dtype for inference",
    )

    # Inference settings
    max_text_length: int = Field(default=500, description="Maximum text length in characters")
    target_rms: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Target RMS for audio normalization",
    )
    cross_fade_duration: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Cross-fade duration between chunks in seconds",
    )
    nfe_step: int = Field(
        default=32,
        ge=4,
        le=128,
        description="Number of function evaluations for ODE solver",
    )
    cfg_strength: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Classifier-free guidance strength",
    )
    sway_sampling_coef: float = Field(
        default=-1.0,
        description="Sway sampling coefficient (-1 to disable)",
    )

    # Audio settings
    sample_rate: int = Field(default=24000, description="Output sample rate")
    hop_length: int = Field(default=256, description="Hop length for vocoder")

    # Voice cloning settings
    clone_min_duration: float = Field(
        default=3.0,
        description="Minimum duration for voice cloning samples (seconds)",
    )
    clone_max_duration: float = Field(
        default=30.0,
        description="Maximum duration for voice cloning samples (seconds)",
    )
    speaker_embedding_dim: int = Field(default=192, description="Speaker embedding dimension")

    # Performance settings
    use_torch_compile: bool = Field(default=False, description="Use torch.compile for optimization")
    warmup_on_load: bool = Field(default=True, description="Run warmup inference on load")

    # Supported languages
    supported_languages: list[str] = Field(
        default=["en", "zh", "ja", "ko", "de", "fr", "es", "it", "ru"],
        description="Supported language codes",
    )

    # Mock mode
    use_mock: bool = Field(default=False, description="Use mock inference for testing")

    # Voices directory
    voices_directory: str = Field(
        default="./voices",
        description="Directory for storing voice profiles",
    )

    model_config = {
        "env_prefix": "F5TTS_",
        "extra": "ignore",
    }

    def get_device(self) -> str:
        """Get the actual device to use."""
        if self.device == "auto":
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def get_dtype(self):
        """Get the torch dtype."""
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.float32)
