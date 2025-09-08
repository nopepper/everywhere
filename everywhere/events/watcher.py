"""Event types."""

from enum import StrEnum
from pathlib import Path

from . import Event


class ChangeType(StrEnum):
    """Event type."""

    ADDED = "added"
    CHANGED = "changed"
    REMOVED = "removed"


class FileChanged(Event):
    """Watch event."""

    path: Path
    event_type: ChangeType
