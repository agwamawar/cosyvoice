"""Timing utilities for performance measurement.

Provides:
- @timed decorator that logs execution time (works with sync and async)
- Timer context manager for timing code blocks
- StopWatch for multi-lap timing
"""

import asyncio
import functools
import time
from collections.abc import Callable, Coroutine
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar

from src.utils.logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TimingResult:
    """Result of a timing measurement."""

    name: str
    duration_seconds: float
    start_time: float
    end_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000

    def __repr__(self) -> str:
        return f"TimingResult(name={self.name!r}, duration_ms={self.duration_ms:.2f})"


class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("my_operation") as t:
            do_something()
        print(f"Took {t.elapsed_ms}ms")

        # Or without logging
        with Timer("silent_op", log=False) as t:
            do_something()
    """

    def __init__(
        self,
        name: str = "operation",
        log: bool = True,
        log_level: str = "debug",
    ) -> None:
        self.name = name
        self.log = log
        self.log_level = log_level.lower()
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration: float = 0
        self._running = False

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        self._running = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        self._running = False

        if self.log:
            log_fn = getattr(logger, self.log_level, logger.debug)
            if exc_type is not None:
                log_fn(
                    f"{self.name} failed",
                    duration_ms=round(self.duration * 1000, 2),
                    error=str(exc_val),
                )
            else:
                log_fn(
                    f"{self.name} completed",
                    duration_ms=round(self.duration * 1000, 2),
                )

    async def __aenter__(self) -> "Timer":
        return self.__enter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if not self._running:
            return self.duration
        return time.perf_counter() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return self.elapsed * 1000

    def result(self) -> TimingResult:
        """Get timing result."""
        return TimingResult(
            name=self.name,
            duration_seconds=self.duration,
            start_time=self.start_time,
            end_time=self.end_time,
        )


@contextmanager
def timed(name: str = "operation", log: bool = True, log_level: str = "debug"):
    """
    Context manager for timing operations (sync version).

    Usage:
        with timed("database_query"):
            result = db.query(...)
    """
    timer = Timer(name, log=log, log_level=log_level)
    with timer:
        yield timer


@asynccontextmanager
async def timed_async_context(
    name: str = "operation",
    log: bool = True,
    log_level: str = "debug",
):
    """
    Async context manager for timing operations.

    Usage:
        async with timed_async_context("api_call"):
            result = await client.get(...)
    """
    timer = Timer(name, log=log, log_level=log_level)
    async with timer:
        yield timer


def timed_sync(
    name: str | None = None,
    log: bool = True,
    log_level: str = "debug",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for timing synchronous functions.

    Usage:
        @timed_sync("process_data")
        def process_data(items):
            ...

        @timed_sync()  # Uses function name
        def another_function():
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        operation_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with Timer(operation_name, log=log, log_level=log_level):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def timed_async(
    name: str | None = None,
    log: bool = True,
    log_level: str = "debug",
) -> Callable[[Callable[P, Coroutine[Any, Any, T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """
    Decorator for timing async functions.

    Usage:
        @timed_async("fetch_data")
        async def fetch_data(url):
            ...

        @timed_async()  # Uses function name
        async def another_async_function():
            ...
    """

    def decorator(
        func: Callable[P, Coroutine[Any, Any, T]]
    ) -> Callable[P, Coroutine[Any, Any, T]]:
        operation_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with Timer(operation_name, log=log, log_level=log_level):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def timed_decorator(
    name: str | None = None,
    log: bool = True,
    log_level: str = "debug",
) -> Callable[[F], F]:
    """
    Universal decorator that works with both sync and async functions.

    Usage:
        @timed_decorator("operation")
        def sync_function():
            ...

        @timed_decorator("operation")
        async def async_function():
            ...
    """

    def decorator(func: F) -> F:
        operation_name = name or func.__name__

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with Timer(operation_name, log=log, log_level=log_level):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with Timer(operation_name, log=log, log_level=log_level):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator


class StopWatch:
    """
    Stopwatch for measuring multiple laps/stages.

    Usage:
        sw = StopWatch()
        sw.start()

        do_step_1()
        sw.lap("step_1")

        do_step_2()
        sw.lap("step_2")

        total = sw.stop()

        for name, duration in sw.get_laps():
            print(f"{name}: {duration*1000:.2f}ms")
    """

    def __init__(self, name: str = "stopwatch") -> None:
        self.name = name
        self._start_time: float | None = None
        self._laps: list[tuple[str, float]] = []  # (name, absolute_time)
        self._running = False

    def start(self) -> "StopWatch":
        """Start the stopwatch."""
        self._start_time = time.perf_counter()
        self._running = True
        self._laps = []
        return self

    def lap(self, name: str = "") -> float:
        """
        Record a lap time.

        Returns:
            Duration of this lap in seconds
        """
        if not self._running or self._start_time is None:
            raise RuntimeError("Stopwatch not started")

        current = time.perf_counter()

        # Calculate duration since last lap (or start)
        if self._laps:
            duration = current - self._laps[-1][1]
        else:
            duration = current - self._start_time

        self._laps.append((name or f"lap_{len(self._laps) + 1}", current))
        return duration

    def stop(self) -> float:
        """
        Stop the stopwatch.

        Returns:
            Total duration in seconds
        """
        if not self._running or self._start_time is None:
            raise RuntimeError("Stopwatch not started")

        self._running = False
        return time.perf_counter() - self._start_time

    def reset(self) -> None:
        """Reset the stopwatch."""
        self._start_time = None
        self._laps = []
        self._running = False

    def get_laps(self) -> list[tuple[str, float]]:
        """
        Get all lap times with durations.

        Returns:
            List of (lap_name, duration_seconds) tuples
        """
        if not self._start_time:
            return []

        results: list[tuple[str, float]] = []
        prev_time = self._start_time

        for name, timestamp in self._laps:
            duration = timestamp - prev_time
            results.append((name, duration))
            prev_time = timestamp

        return results

    def get_laps_ms(self) -> list[tuple[str, float]]:
        """Get all lap times in milliseconds."""
        return [(name, duration * 1000) for name, duration in self.get_laps()]

    @property
    def total(self) -> float:
        """Get total elapsed time in seconds."""
        if self._start_time is None:
            return 0.0

        if self._running:
            return time.perf_counter() - self._start_time

        if self._laps:
            return self._laps[-1][1] - self._start_time

        return 0.0

    @property
    def total_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return self.total * 1000

    def log_summary(self, level: str = "info") -> None:
        """Log a summary of all laps."""
        log_fn = getattr(logger, level.lower(), logger.info)
        laps = self.get_laps_ms()

        log_fn(
            f"StopWatch '{self.name}' summary",
            total_ms=round(self.total_ms, 2),
            laps={name: round(dur, 2) for name, dur in laps},
        )


async def measure_async(
    coro: Coroutine[Any, Any, T],
    name: str = "async_operation",
) -> tuple[T, TimingResult]:
    """
    Measure execution time of a coroutine.

    Usage:
        result, timing = await measure_async(some_async_call(), "api_fetch")
        print(f"Result: {result}, took {timing.duration_ms}ms")

    Returns:
        Tuple of (result, TimingResult)
    """
    start = time.perf_counter()
    result = await coro
    end = time.perf_counter()

    timing = TimingResult(
        name=name,
        duration_seconds=end - start,
        start_time=start,
        end_time=end,
    )

    return result, timing


def measure_sync(
    func: Callable[P, T],
    *args: P.args,
    name: str | None = None,
    **kwargs: P.kwargs,
) -> tuple[T, TimingResult]:
    """
    Measure execution time of a synchronous function call.

    Usage:
        result, timing = measure_sync(expensive_function, arg1, arg2, name="compute")
        print(f"Result: {result}, took {timing.duration_ms}ms")

    Returns:
        Tuple of (result, TimingResult)
    """
    operation_name = name or func.__name__
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()

    timing = TimingResult(
        name=operation_name,
        duration_seconds=end - start,
        start_time=start,
        end_time=end,
    )

    return result, timing
