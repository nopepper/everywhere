"""Simple aggregated search provider."""

from collections.abc import Iterable
from typing import TypeVar

from pydantic import Field

from .search_provider import SearchProvider, SearchQuery, SearchResult

V = TypeVar("V")


class AggregatedSearchProvider(SearchProvider[str, V]):
    """Aggregated search provider."""

    search_providers: list[SearchProvider[str, V]] = Field(description="Search providers to aggregate.")

    def search(self, query: SearchQuery[str]) -> Iterable[SearchResult[V]]:
        """Search for a query."""
        return [result for provider in self.search_providers for result in provider.search(query)]
