"""Clock interface for dependency injection."""

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable


class Timer:
    """Timer interface."""

    def __init__(self, delay: float, callback: Callable[[], None]):
        """Store timer configuration for later execution."""
        self.delay = delay
        self.callback = callback
        self._cancelled = False

    def cancel(self) -> None:
        """Cancel the timer."""
        self._cancelled = True

    def start(self) -> None:
        """Start the timer."""
        if not self._cancelled:
            self.callback()


class Clock(ABC):
    """Clock interface for creating timers."""

    @abstractmethod
    def timer(self, delay: float, callback: Callable[[], None]) -> Timer:
        """Create a timer that will call callback after delay seconds."""


class ThreadingTimer(Timer):
    """Timer implementation using threading.Timer."""

    def __init__(self, delay: float, callback: Callable[[], None]):
        """Wrap a ``threading.Timer`` to defer execution."""
        super().__init__(delay, callback)
        self._timer = threading.Timer(delay, callback)

    def cancel(self) -> None:
        """Cancel the timer."""
        super().cancel()
        self._timer.cancel()

    def start(self) -> None:
        """Start the timer."""
        if not self._cancelled:
            self._timer.start()


class ThreadingClock(Clock):
    """Real clock implementation using threading.Timer."""

    def timer(self, delay: float, callback: Callable[[], None]) -> Timer:
        """Create a timer that will call callback after delay seconds."""
        return ThreadingTimer(delay, callback)
