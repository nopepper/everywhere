"""Search provider events."""

from pathlib import Path

from . import Event


class ProviderEvent(Event):
    """Provider event."""


class IndexingStarted(ProviderEvent):
    """Indexing started event."""

    path: Path


class IndexingFinished(ProviderEvent):
    """Indexing finished event."""

    success: bool
    path: Path
