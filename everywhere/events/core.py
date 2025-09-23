"""A simple, thread-safe event bus implementation using weak references.

This module provides a decoupled way for components to communicate through events.
It supports:
- Publishing events to subscribers.
- Subscribing to events with callbacks that run in dedicated worker threads.
- Subscribing to events with queue-based listeners for manual processing.
- Automatic garbage collection of subscriptions when subscribers are no longer in use.
- Correlation IDs to trace event flows through the system.
"""

import contextvars
import functools
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Generator
from typing import Any

from pydantic import Field

from ..common.pydantic import FrozenBaseModel

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id")


def _get_correlation_id() -> str:
    """Get the current correlation ID, or generate a new one if not set.

    Returns:
        The correlation ID for the current context.
    """
    try:
        return _correlation_id.get()
    except LookupError:
        return str(uuid.uuid4())


def correlated(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to ensure a function executes within a correlated context.

    If a correlation ID is not present in the context, a new one is generated
    and set for the duration of the function call.

    Args:
        func: The function to wrap.

    Returns:
        The wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            _correlation_id.get()
            return func(*args, **kwargs)
        except LookupError:
            token = _correlation_id.set(str(uuid.uuid4()))
            try:
                return func(*args, **kwargs)
            finally:
                _correlation_id.reset(token)

    return wrapper


class Event(FrozenBaseModel):
    """Base class for all events, providing a default correlation ID."""

    event_t: int = Field(default_factory=time.time_ns)
    correlation_id: str = Field(default_factory=_get_correlation_id)


class EventBus:
    """Event bus implementation."""

    def __init__(self, history_size: int) -> None:
        """Initialize the event bus."""
        self._events = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._ping = threading.Condition()

    def submit_event(self, event: Event) -> None:
        """Submit an event to the event bus."""
        with self._lock, self._ping:
            self._events.append(event)
            self._ping.notify_all()

    def get_events(
        self, event_type: type[Event] | type[Any] | None, after_t: int | None = None, limit: int | None = None
    ) -> list[Event]:
        """Get events from the event bus."""
        result = []
        with self._lock:
            if len(self._events) == 0:
                return []
            for i in range(len(self._events), 0, -1):
                ev = self._events[i - 1]
                time_ok = after_t is None or ev.event_t > after_t
                type_ok = event_type is None or isinstance(ev, event_type)
                if time_ok:
                    if type_ok:
                        result.append(ev)
                        if limit is not None and len(result) >= limit:
                            break
                else:
                    break
        return result[::-1]

    def wait_for_event(self, event_type: type[Event] | None = None) -> None:
        """Wait for an event to be published."""
        while True:
            with self._ping:
                self._ping.wait()
            if event_type is not None:
                with self._lock:
                    if isinstance(self._events[-1], event_type):
                        break
            else:
                break

    def get_last_event(self, event_type: type[Event] | None, after_t: int | None = None) -> Event:
        """Get the last event from the event bus."""
        result = self.get_events(event_type=event_type, after_t=after_t, limit=1)
        while len(result) == 0:
            self.wait_for_event(event_type)
            result = self.get_events(event_type=event_type, after_t=after_t, limit=1)
        return result[0]

    def stream_windows(
        self, event_type: type[Event] | None, window_time: float = 0.1
    ) -> Generator[list[Event], None, None]:
        """Wait for an event, then yield a window of subsequent events.

        This generator waits for an event of `event_type`. Once received, it waits
        for `window_time` seconds and then yields all events that occurred after
        the initial trigger event. This process repeats indefinitely.

        Args:
            event_type: The type of event to wait for to start a window.
            window_time: The duration to wait before collecting events, in seconds.

        Yields:
            A list of events that occurred within the window after the trigger.
        """
        after_t: int | None = None
        while True:
            trigger_event = self.get_last_event(event_type=event_type, after_t=after_t)
            start_of_window_t = trigger_event.event_t
            time.sleep(window_time)
            events_in_window = [trigger_event, *self.get_events(event_type=None, after_t=start_of_window_t)]
            after_t = events_in_window[-1].event_t
            yield events_in_window
