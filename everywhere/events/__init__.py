"""Simple event bus for tracking app activity."""

import contextvars
import functools
import threading
from collections import defaultdict
from collections.abc import Callable
from queue import SimpleQueue
from typing import Any, TypeVar
import uuid

from pydantic import Field
from ..common.pydantic import FrozenBaseModel

# Context variable for correlation ID
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id")


def _get_correlation_id() -> str:
    """Get correlation ID from context or generate a new one."""
    try:
        return _correlation_id.get()
    except LookupError:
        return str(uuid.uuid4())


class Event(FrozenBaseModel):
    """Event type."""

    correlation_id: str = Field(default_factory=_get_correlation_id, description="Correlation ID for the event.")


_listeners: dict[type[Event] | None, list[SimpleQueue]] = defaultdict(list)
_main_queue: SimpleQueue = SimpleQueue()

T = TypeVar("T", bound=Event)


def _process_events():
    while True:
        event = _main_queue.get()
        for listener in _listeners[None]:
            listener.put(event)
        for listener in _listeners[type(event)]:
            listener.put(event)


def get_listener(event_type: type[Event] | None = None) -> SimpleQueue:
    """Create a listener for a specific event type."""
    queue = SimpleQueue()
    _listeners[event_type].append(queue)
    return queue


def publish(event: Event):
    """Publish an event."""
    _main_queue.put(event)


def add_callback(event_type: type[T], callback: Callable[[T], None]):
    """Add a callback to a listener."""
    listener = get_listener(event_type)

    def callback_func():
        while True:
            event = listener.get()
            callback(event)

    threading.Thread(target=callback_func, daemon=True).start()


def correlated(func: Callable) -> Callable:
    """Correlated decorator."""

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


threading.Thread(target=_process_events, daemon=True).start()
