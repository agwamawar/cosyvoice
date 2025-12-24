"""Configuration module for TTS Backend."""

from src.config.settings import (
    APISettings,
    AppSettings,
    AudioSettings,
    ModelSettings,
    ServerSettings,
    Settings,
    get_settings,
)

__all__ = [
    "AppSettings",
    "ServerSettings",
    "ModelSettings",
    "AudioSettings",
    "APISettings",
    "Settings",
    "get_settings",
]
