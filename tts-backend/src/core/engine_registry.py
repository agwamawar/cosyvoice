"""Thread-safe registry for managing TTS engines.

Provides:
- Singleton registry for engine registration and access
- Lazy loading of engines on first access
- Thread-safe operations
"""

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

from src.core.interfaces import TTSEngine
from src.utils.logging import get_logger

logger = get_logger("system")


@dataclass
class EngineRegistration:
    """Registration info for a TTS engine."""

    engine_class: type[TTSEngine]
    config: dict[str, Any] = field(default_factory=dict)
    description: str = ""


class EngineRegistry:
    """
    Thread-safe singleton registry for TTS engines.

    Usage:
        # Register an engine class
        registry = EngineRegistry()
        registry.register("cosyvoice", CosyVoiceEngine, config={"model_path": "..."})

        # Get an engine instance (lazy loads on first access)
        engine = await registry.get_engine("cosyvoice")

        # Use the engine
        result = await engine.synthesize(text, options)
    """

    _instance: "EngineRegistry | None" = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EngineRegistry":
        """Singleton pattern - return existing instance or create new one."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the registry (only runs once due to singleton)."""
        if self._initialized:
            return

        self._registrations: dict[str, EngineRegistration] = {}
        self._instances: dict[str, TTSEngine] = {}
        self._loading_locks: dict[str, asyncio.Lock] = {}
        self._default_engine: str | None = None
        self._mutex = threading.RLock()
        self._initialized = True

        logger.debug("EngineRegistry initialized")

    def register(
        self,
        name: str,
        engine_class: type[TTSEngine],
        config: dict[str, Any] | None = None,
        description: str = "",
        set_default: bool = False,
    ) -> None:
        """
        Register a TTS engine class.

        Args:
            name: Unique name for the engine
            engine_class: Engine class (must inherit from TTSEngine)
            config: Configuration to pass to engine constructor
            description: Optional description of the engine
            set_default: Whether to set this as the default engine
        """
        with self._mutex:
            if name in self._registrations:
                logger.warning(f"Overwriting existing engine registration: {name}")

            self._registrations[name] = EngineRegistration(
                engine_class=engine_class,
                config=config or {},
                description=description,
            )

            if set_default or self._default_engine is None:
                self._default_engine = name

            logger.info(
                f"Registered TTS engine: {name}",
                engine_class=engine_class.__name__,
                is_default=(self._default_engine == name),
            )

    def register_decorator(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        set_default: bool = False,
    ):
        """
        Decorator to register a TTS engine class.

        Usage:
            @registry.register_decorator("my_engine")
            class MyEngine(TTSEngine):
                ...
        """

        def decorator(engine_class: type[TTSEngine]) -> type[TTSEngine]:
            self.register(name, engine_class, config, set_default=set_default)
            return engine_class

        return decorator

    async def get_engine(self, name: str | None = None) -> TTSEngine:
        """
        Get a TTS engine instance by name (lazy loads on first access).

        Args:
            name: Engine name. If None, returns the default engine.

        Returns:
            Loaded TTSEngine instance

        Raises:
            ValueError: If engine name is not registered
            ModelLoadError: If engine fails to load
        """
        engine_name = name or self._default_engine

        if engine_name is None:
            raise ValueError("No engine specified and no default engine set")

        # Check if already loaded
        with self._mutex:
            if engine_name in self._instances:
                return self._instances[engine_name]

            if engine_name not in self._registrations:
                raise ValueError(
                    f"Unknown engine: {engine_name}. "
                    f"Available engines: {list(self._registrations.keys())}"
                )

            # Create async lock for this engine if not exists
            if engine_name not in self._loading_locks:
                self._loading_locks[engine_name] = asyncio.Lock()

        # Load engine with async lock (prevents duplicate loading)
        async with self._loading_locks[engine_name]:
            # Double-check after acquiring lock
            if engine_name in self._instances:
                return self._instances[engine_name]

            logger.info(f"Loading TTS engine: {engine_name}")
            engine = await self._load_engine(engine_name)

            with self._mutex:
                self._instances[engine_name] = engine

            logger.info(f"TTS engine loaded: {engine_name}")
            return engine

    async def _load_engine(self, name: str) -> TTSEngine:
        """Load and initialize an engine instance."""
        registration = self._registrations[name]

        # Create engine instance
        engine = registration.engine_class(**registration.config)

        # Load the model
        await engine.load()

        return engine

    def get_engine_sync(self, name: str | None = None) -> TTSEngine | None:
        """
        Get an already-loaded engine instance (non-async, no loading).

        Returns None if the engine is not yet loaded.
        """
        engine_name = name or self._default_engine
        if engine_name is None:
            return None

        with self._mutex:
            return self._instances.get(engine_name)

    def list_engines(self) -> list[str]:
        """List all registered engine names."""
        with self._mutex:
            return list(self._registrations.keys())

    def list_loaded_engines(self) -> list[str]:
        """List all currently loaded engine names."""
        with self._mutex:
            return list(self._instances.keys())

    def get_default_engine_name(self) -> str | None:
        """Get the name of the default engine."""
        return self._default_engine

    def set_default_engine(self, name: str) -> None:
        """
        Set the default engine.

        Args:
            name: Engine name (must be registered)

        Raises:
            ValueError: If engine is not registered
        """
        with self._mutex:
            if name not in self._registrations:
                raise ValueError(f"Unknown engine: {name}")
            self._default_engine = name
            logger.info(f"Default engine set to: {name}")

    def is_registered(self, name: str) -> bool:
        """Check if an engine is registered."""
        with self._mutex:
            return name in self._registrations

    def is_loaded(self, name: str) -> bool:
        """Check if an engine is loaded and ready."""
        with self._mutex:
            if name not in self._instances:
                return False
            return self._instances[name].is_loaded

    def get_registration(self, name: str) -> EngineRegistration | None:
        """Get registration info for an engine."""
        with self._mutex:
            return self._registrations.get(name)

    async def unload_engine(self, name: str) -> None:
        """
        Unload a specific engine.

        Args:
            name: Engine name to unload
        """
        with self._mutex:
            if name not in self._instances:
                return
            engine = self._instances.pop(name)

        logger.info(f"Unloading TTS engine: {name}")
        await engine.unload()
        logger.info(f"TTS engine unloaded: {name}")

    async def unload_all(self) -> None:
        """Unload all loaded engines."""
        with self._mutex:
            engine_names = list(self._instances.keys())

        for name in engine_names:
            await self.unload_engine(name)

    async def reload_engine(self, name: str) -> TTSEngine:
        """
        Reload an engine (unload then load).

        Args:
            name: Engine name to reload

        Returns:
            Reloaded engine instance
        """
        await self.unload_engine(name)
        return await self.get_engine(name)

    def unregister(self, name: str) -> None:
        """
        Unregister an engine (must be unloaded first).

        Args:
            name: Engine name to unregister

        Raises:
            RuntimeError: If engine is still loaded
        """
        with self._mutex:
            if name in self._instances:
                raise RuntimeError(
                    f"Cannot unregister loaded engine: {name}. Unload it first."
                )

            if name in self._registrations:
                del self._registrations[name]
                logger.info(f"Unregistered TTS engine: {name}")

            if self._default_engine == name:
                # Set new default if available
                self._default_engine = next(iter(self._registrations.keys()), None)

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._registrations.clear()
                cls._instance._instances.clear()
                cls._instance._loading_locks.clear()
                cls._instance._default_engine = None


# Global registry instance
_registry: EngineRegistry | None = None


def get_engine_registry() -> EngineRegistry:
    """Get the global engine registry instance."""
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry
