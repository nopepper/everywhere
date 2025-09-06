"""Simple aggregated search provider."""

from collections.abc import Iterable

from pydantic import Field

from ..common.pydantic import SearchQuery, SearchResult, WatchEvent
from ..watchers.fs_watcher import FSWatcher
from .search_provider import SearchProvider


class FSSearchProvider(SearchProvider):
    """Aggregated search provider. Only supports text for now."""

    search_providers: list[SearchProvider] = Field(description="Search providers to aggregate.")
    watcher: FSWatcher = Field(description="Watcher for the database.")

    def on_change(self, event: WatchEvent) -> None:
        """Handle a change event."""
        for provider in self.search_providers:
            provider.on_change(event)

    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""
        results: list[SearchResult] = []
        for provider in self.search_providers:
            results.extend(provider.search(query))
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def setup(self) -> None:
        """Setup the provider."""
        for provider in self.search_providers:
            provider.setup()
        for event in self.watcher.update():
            self.on_change(event)

    def teardown(self) -> None:
        """Teardown the provider."""
        for provider in self.search_providers:
            provider.teardown()
