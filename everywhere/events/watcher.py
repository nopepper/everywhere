"""Event types."""

from enum import StrEnum
from pathlib import Path

from . import Event


class ChangeType(StrEnum):
    """Event type."""

    UPSERT = "upsert"
    REMOVE = "remove"


class FileChanged(Event):
    """Watch event."""

    path: Path
    event_type: ChangeType
