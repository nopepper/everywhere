"""Basic search provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable

from pydantic import BaseModel

from ..common.pydantic import SearchQuery, SearchResult


class SearchProvider(BaseModel, ABC):
    """Search provider."""

    @property
    @abstractmethod
    def supported_types(self) -> set[str]:
        """Supported file types."""

    @abstractmethod
    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""
