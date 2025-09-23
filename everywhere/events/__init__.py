"""Events."""

import threading
import time
from typing import Any, TypeVar, cast
from collections.abc import Generator
from collections.abc import Callable
from .core import Event, EventBus, correlated

HISTORY_SIZE = 1_000_000

_default_event_bus = EventBus(HISTORY_SIZE)


def publish(event: Event) -> None:
    """Publish an event."""
    return _default_event_bus.submit_event(event)


def get_events(
    event_type: type[Event] | type[Any] | None, after_t: int | None = None, limit: int | None = None
) -> list[Event]:
    """Get events from the event bus."""
    return _default_event_bus.get_events(event_type=event_type, after_t=after_t, limit=limit)


def get_last_event(event_type: type[Event] | type[Any] | None, after_t: int | None = None) -> Event:
    """Get the next event from the event bus."""
    return _default_event_bus.get_last_event(event_type=event_type, after_t=after_t)


def wait_for_event(event_type: type[Event] | None = None) -> None:
    """Wait for an event to be published."""
    _default_event_bus.wait_for_event(event_type=event_type)


def stream_windows(event_type: type[Event] | None, window_time: float = 0.1) -> Generator[list[Event], None, None]:
    """Stream windows of events from the event bus."""
    return _default_event_bus.stream_windows(event_type=event_type, window_time=window_time)


T = TypeVar("T")


def add_callback(event_type: type[T] | None, callback: Callable[[T], Any]) -> None:
    """Register a callback for a specific event type."""

    def worker() -> None:
        t_prev = time.time_ns()
        while True:
            wait_for_event()
            events = get_events(event_type, t_prev)
            if not events:
                continue
            for event in events:
                callback(cast("T", event))
            t_prev = events[-1].event_t

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
