"""Configuration for CosyVoice model."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class CosyVoiceConfig(BaseSettings):
    """Configuration for CosyVoice TTS model."""

    # Model paths
    model_path: str = Field(
        default="/workspace/models/Fun-CosyVoice3-0.5B",
        description="Path to model directory",
    )
    llm_model: str = Field(default="llm.pt", description="LLM model filename")
    flow_model: str = Field(default="flow.pt", description="Flow model filename")
    hift_model: str = Field(default="hift.pt", description="HiFT vocoder filename")

    # Device settings
    device: Literal["cuda", "cpu", "auto"] = Field(
        default="auto",
        description="Device to run inference on (cuda, cpu, or auto)",
    )
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float16",
        description="Model dtype for inference",
    )

    # Inference settings
    max_text_length: int = Field(default=500, description="Maximum text length in characters")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)

    # Audio settings
    sample_rate: int = Field(default=22050, description="Output sample rate")
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
    clone_sample_rate: int = Field(
        default=16000,
        description="Sample rate for voice cloning input",
    )
    speaker_embedding_dim: int = Field(default=192, description="Speaker embedding dimension")

    # Performance settings
    use_torch_compile: bool = Field(default=False, description="Use torch.compile for optimization")
    compile_mode: str = Field(default="reduce-overhead", description="torch.compile mode")
    warmup_on_load: bool = Field(default=True, description="Run warmup inference on load")

    # Supported languages
    supported_languages: list[str] = Field(
        default=["en", "zh", "ja", "ko", "de", "fr", "es"],
        description="Supported language codes",
    )

    # Mock mode
    use_mock: bool = Field(default=False, description="Use mock inference for testing")

    model_config = {
        "env_prefix": "COSYVOICE_",
        "extra": "ignore",
    }

    def get_model_paths(self) -> dict[str, str]:
        """Get full paths to model files."""
        base = Path(self.model_path)
        return {
            "llm": str(base / self.llm_model),
            "flow": str(base / self.flow_model),
            "hift": str(base / self.hift_model),
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
