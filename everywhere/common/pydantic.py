"""Pydantic base model."""

from enum import StrEnum
from pathlib import Path
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

ValueType: TypeAlias = Path


class FrozenBaseModel(BaseModel):
    """Pydantic frozen base model."""

    model_config = ConfigDict(frozen=True, strict=True)


class EventType(StrEnum):
    """Event type."""

    ADDED = "added"
    CHANGED = "changed"
    REMOVED = "removed"


class WatchEvent(FrozenBaseModel):
    """Watch event."""

    value: ValueType
    event_type: EventType


class SearchQuery(FrozenBaseModel):
    """Search query."""

    text: str


class SearchResult(FrozenBaseModel):
    """Search result."""

    value: ValueType
    confidence: float
