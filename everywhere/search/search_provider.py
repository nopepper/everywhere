"""Basic search provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Self

from ..common.pydantic import SearchQuery, SearchResult
from ..events.watcher import FileChanged


class SearchProvider(ABC):
    """Search provider service."""

    @abstractmethod
    def __enter__(self) -> Self:
        """Start the search provider."""

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Stop the search provider."""

    @abstractmethod
    def update_index(self, event: FileChanged) -> None:
        """Handle a file changed event."""

    @abstractmethod
    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""
