"""Basic search provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..common.pydantic import FrozenBaseModel, SearchQuery, SearchResult, WatchEvent


class SearchProvider(FrozenBaseModel, ABC):
    """Search provider."""

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """Supported document types."""

    @abstractmethod
    def on_change(self, event: WatchEvent) -> None:
        """Handle a change event."""

    @abstractmethod
    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""

    def setup(self) -> None:
        """Setup the provider."""

    def teardown(self) -> None:
        """Teardown the provider."""
