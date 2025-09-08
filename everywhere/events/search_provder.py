"""Search provider events."""

from pathlib import Path

from . import Event


class ProviderEvent(Event):
    """Provider event."""


class SetupStarted(ProviderEvent):
    """Setup started event."""


class SetupFinished(ProviderEvent):
    """Setup finished event."""


class TeardownStarted(ProviderEvent):
    """Teardown started event."""


class TeardownFinished(ProviderEvent):
    """Teardown finished event."""


class IndexingStarted(ProviderEvent):
    """Indexing started event."""

    path: Path


class IndexingFinished(ProviderEvent):
    """Indexing finished event."""

    success: bool
    path: Path


class SearchStarted(ProviderEvent):
    """Search started event."""


class SearchFinished(ProviderEvent):
    """Search finished event."""

    success: bool
    result_count: int
