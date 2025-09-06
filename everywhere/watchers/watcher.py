"""Basic watcher interface."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar


class EventType(StrEnum):
    """Event type."""

    ADDED = "added"
    CHANGED = "changed"
    REMOVED = "deleted"


V = TypeVar("V")


@dataclass(frozen=True, slots=True)
class WatchEvent(Generic[V]):
    """Watch event."""

    value: V
    event_type: EventType
