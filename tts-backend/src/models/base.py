"""Base class for TTS models."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseTTSModel(ABC):
    """Abstract base class for TTS model wrappers."""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.device = device
        self.dtype = dtype
        self._model: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load the model from the specified path."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""
        ...

    @abstractmethod
    def generate(
        self,
        text: str,
        voice_embedding: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Generate audio from text.

        Args:
            text: Input text to synthesize
            voice_embedding: Optional voice embedding for voice cloning
            **kwargs: Additional generation parameters

        Returns:
            Audio tensor
        """
        ...

    def to(self, device: str) -> "BaseTTSModel":
        """Move model to specified device."""
        self.device = device
        if self._model is not None:
            self._model = self._model.to(device)
        return self

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
