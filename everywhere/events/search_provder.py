"""Search provider events."""

from pathlib import Path

from ..common.pydantic import SearchResult
from . import Event


class ProviderEvent(Event):
    """Provider event."""


class SearchStarted(ProviderEvent):
    """Search started event."""

    query: str


class GotSearchResult(ProviderEvent):
    """Got search result event."""

    query: str
    result: SearchResult


class SearchFinished(ProviderEvent):
    """Search finished event."""

    query: str


class GotIndexingRequest(ProviderEvent):
    """Got indexing request event."""

    path: Path


class IndexingStarted(ProviderEvent):
    """Indexing started event."""

    path: Path


class IndexingFinished(ProviderEvent):
    """Indexing finished event."""

    success: bool
    path: Path
