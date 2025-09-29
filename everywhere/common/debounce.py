"""Debounced runner."""

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any


class AsyncDebouncedRunner:
    """Debounced runner."""

    def __init__(self, delay: float):
        """Initialize the debounced runner."""
        self._delay = delay
        self._task: asyncio.Task | None = None

    def cancel(self) -> None:
        """Cancel the debounced runner."""
        if self._task:
            self._task.cancel()
        self._task = None

    async def _run_func(self, func: Callable[[], Any | Awaitable[Any]]) -> Any:
        """Run the function."""
        await asyncio.sleep(self._delay)
        result = func()
        if isinstance(result, Awaitable):
            return await result
        else:
            return result

    def submit(self, func: Callable[[], Any | Awaitable[Any]]) -> asyncio.Task[Any]:
        """Run the function."""
        self.cancel()
        self._task = asyncio.create_task(self._run_func(func))
        return self._task


class DebouncedRunner:
    """Debounced runner."""

    def __init__(self, delay: float):
        """Initialize the debounced runner."""
        self._delay = delay
        self._timer: threading.Timer | None = None

    def cancel(self) -> None:
        """Cancel the debounced runner."""
        if self._timer:
            self._timer.cancel()
        self._timer = None

    def submit(self, func: Callable[[], Any]) -> None:
        """Run the function."""
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self._delay, func)
        self._timer.start()
