"""Basic search provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..common.pydantic import SearchQuery, SearchResult
from ..events.watcher import FileChanged


class SearchProviderService(ABC):
    """Search provider service."""

    @abstractmethod
    def start(self) -> None:
        """Start the search provider."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the search provider."""

    @abstractmethod
    def handle_file_change(self, event: FileChanged) -> None:
        """Handle a file changed event."""

    @abstractmethod
    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""
