"""Simple event bus for tracking app activity."""

import threading
from collections import defaultdict
from collections.abc import Callable
from queue import SimpleQueue
from typing import TypeVar

from pydantic import BaseModel

_listeners: dict[type[BaseModel] | None, list[SimpleQueue]] = defaultdict(list)
_main_queue: SimpleQueue = SimpleQueue()

T = TypeVar("T", bound=BaseModel)


def _process_events():
    while True:
        event = _main_queue.get()
        for listener in _listeners[None]:
            listener.put(event)
        for listener in _listeners[type(event)]:
            listener.put(event)


def get_listener(event_type: type[BaseModel] | None = None) -> SimpleQueue:
    """Create a listener for a specific event type."""
    queue = SimpleQueue()
    _listeners[event_type].append(queue)
    return queue


def publish(event: BaseModel):
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


threading.Thread(target=_process_events, daemon=True).start()
