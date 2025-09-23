"""Debounced runner."""

import threading
from collections.abc import Callable
from typing import Any


class DebouncedRunner:
    """Debounced runner."""

    def __init__(self, delay: float):
        """Initialize the debounced runner."""
        self._delay = delay
        self._timer: threading.Timer | None = None

    def submit(self, func: Callable[[], Any]) -> None:
        """Run the function."""
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self._delay, func)
        self._timer.start()
