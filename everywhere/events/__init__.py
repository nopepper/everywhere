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
import inspect
import threading
import uuid
import weakref
from collections import defaultdict
from queue import Empty, SimpleQueue
from typing import Any, TypeVar

from pydantic import Field

from ..common.pydantic import FrozenBaseModel
from collections.abc import Callable


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


class Event(FrozenBaseModel):
    """Base class for all events, providing a default correlation ID."""

    correlation_id: str = Field(default_factory=_get_correlation_id)


class _QueueListener:
    """A queue wrapper that can be weak-referenced."""

    __slots__ = ("__weakref__", "queue")

    def __init__(self) -> None:
        self.queue: SimpleQueue[Any] = SimpleQueue()

    def get(self, timeout: float | None = None) -> Any:
        if timeout is None:
            return self.queue.get()
        return self.queue.get(timeout=timeout)

    def put(self, item: Any) -> None:
        self.queue.put(item)

    def close(self) -> None:
        self.queue.put(None)


T = TypeVar("T", bound=Event)


_lock = threading.RLock()
_subscriptions: dict[type[Event] | None, set[weakref.ReferenceType[_QueueListener]]] = defaultdict(set)


def _prune_dead_subscribers(subscribers: set[weakref.ReferenceType[_QueueListener]]) -> None:
    """Remove dead weak references from a set of subscribers.

    Args:
        subscribers: A set of weak references to `_QueueListener` instances.
    """
    dead_refs = {ref for ref in subscribers if ref() is None}
    if dead_refs:
        subscribers.difference_update(dead_refs)


def get_listener(event_type: type[Event] | None = None) -> _QueueListener:
    """Create and subscribe a queue-like listener for a given event type.

    The subscription remains active as long as a strong reference to the returned
    listener object is maintained.

    Args:
        event_type: The class of the event to listen for. If None, all events
                    will be received.

    Returns:
        A `_QueueListener` instance subscribed to the specified event type.
    """
    listener = _QueueListener()
    with _lock:
        subscribers = _subscriptions[event_type]
        _prune_dead_subscribers(subscribers)
        subscribers.add(weakref.ref(listener))
    return listener


def add_callback(event_type: type[T] | None, callback: Callable[[T], None]) -> Callable[[], None]:
    """Register a callback for a specific event type.

    The callback will execute in a dedicated worker thread. The subscription is
    automatically managed and will be removed when the callback's owner is
    garbage-collected.

    Args:
        event_type: The event class to subscribe to. If None, the callback will
                    be triggered for all events.
        callback: The function or method to execute when an event is published.

    Returns:
        An `unsubscribe` function to manually terminate the subscription and stop
        the worker thread.
    """
    listener = _QueueListener()
    with _lock:
        subscribers = _subscriptions[event_type]
        _prune_dead_subscribers(subscribers)
        listener_ref = weakref.ref(listener)
        subscribers.add(listener_ref)

    callback_ref = weakref.WeakMethod(callback) if inspect.ismethod(callback) else weakref.ref(callback)
    stop_flag = threading.Event()

    def worker() -> None:
        while not stop_flag.is_set():
            resolved_callback = callback_ref()
            if resolved_callback is None:
                break
            try:
                event = listener.get(timeout=1.0)
                if event is None:
                    break
                resolved_callback(event)
            except Empty:
                continue

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def _unsubscribe_core() -> None:
        stop_flag.set()
        listener.close()
        with _lock:
            if event_type in _subscriptions:
                _subscriptions[event_type].discard(listener_ref)

    # Automatically unsubscribe when the callback's owner is garbage-collected.
    # For bound methods, this is the instance; for functions, the function object itself.
    target_obj = callback.__self__ if inspect.ismethod(callback) else callback
    weakref.finalize(target_obj, _unsubscribe_core)

    return _unsubscribe_core


def publish(event: Event) -> None:
    """Publish an event, delivering it to all subscribed listeners.

    Args:
        event: The event object to be published.
    """
    event_type = type(event)
    with _lock:
        # Get subscribers for this specific event type and for all event types (None).
        specific_subscribers = _subscriptions.get(event_type, set())
        all_event_subscribers = _subscriptions.get(None, set())
        target_subscribers = specific_subscribers | all_event_subscribers

    dead_refs: list[weakref.ReferenceType[_QueueListener]] = []
    for ref in target_subscribers:
        listener = ref()
        if listener:
            listener.put(event)
        else:
            dead_refs.append(ref)

    if dead_refs:
        with _lock:
            for key in (None, event_type):
                if key in _subscriptions:
                    _subscriptions[key].difference_update(dead_refs)


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
