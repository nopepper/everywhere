"""Event types."""

from enum import StrEnum
from pathlib import Path

from ..common.pydantic import FrozenBaseModel


class ChangeType(StrEnum):
    """Event type."""

    ADDED = "added"
    CHANGED = "changed"
    REMOVED = "removed"


class FileChangeEvent(FrozenBaseModel):
    """Watch event."""

    path: Path
    event_type: ChangeType
