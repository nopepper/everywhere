"""Simple aggregated search provider."""

from collections.abc import Iterable

from more_itertools import flatten
from pydantic import Field
from tqdm.auto import tqdm

from ..common.pydantic import SearchQuery, SearchResult, WatchEvent
from ..watchers.fs_watcher import FSWatcher
from .search_provider import SearchProvider


class FSSearchProvider(SearchProvider):
    """Aggregated search provider. Only supports text for now."""

    search_providers: list[SearchProvider] = Field(description="Search providers to aggregate.")
    watcher: FSWatcher = Field(description="Watcher for the database.")
    confidence_threshold: float = Field(default=0.0, description="Confidence threshold for the search results.")

    @property
    def supported_types(self) -> list[str]:
        """Supported document types."""
        return list(set(flatten(provider.supported_types for provider in self.search_providers)))

    def on_change(self, event: WatchEvent) -> None:
        """Handle a change event."""
        for provider in self.search_providers:
            provider.on_change(event)

    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""
        results: list[SearchResult] = []
        for provider in self.search_providers:
            results.extend(provider.search(query))
        results = [result for result in results if result.confidence >= self.confidence_threshold]
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def setup(self) -> None:
        """Setup the provider."""
        for provider in self.search_providers:
            provider.setup()
        for event in tqdm(list(self.watcher.update(supported_types=self.supported_types)), desc="Updating database"):
            self.on_change(event)

    def teardown(self) -> None:
        """Teardown the provider."""
        for provider in self.search_providers:
            provider.teardown()
