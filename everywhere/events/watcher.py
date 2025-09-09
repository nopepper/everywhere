"""Event types."""

from enum import StrEnum
from pathlib import Path

from . import Event


class ChangeType(StrEnum):
    """Event type."""

    UPSERT = "upsert"
    DELETE = "delete"


class FileChanged(Event):
    """Watch event."""

    path: Path
    event_type: ChangeType
