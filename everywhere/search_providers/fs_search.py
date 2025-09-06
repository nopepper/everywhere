"""Simple aggregated search provider."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import Field

from ..watchers.fs_watcher import FSWatcher
from ..watchers.watcher import WatchEvent
from .search_provider import SearchProvider, SearchQuery, SearchResult


class FSSearchProvider(SearchProvider[str, Path]):
    """Aggregated search provider. Only supports text for now."""

    text_handlers: list[SearchProvider[str, Path]] = Field(description="Search providers to aggregate.")
    watcher: FSWatcher = Field(description="Watcher for the database.")

    def on_change(self, event: WatchEvent[Path]) -> None:
        """Handle a change event."""
        for provider in self.text_handlers:
            provider.on_change(event)

    def search(self, query: SearchQuery[str]) -> Iterable[SearchResult[Path]]:
        """Search for a query."""
        results: list[SearchResult[Path]] = []
        for provider in self.text_handlers:
            results.extend(provider.search(query))
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def setup(self) -> None:
        """Setup the provider."""
        for provider in self.text_handlers:
            provider.setup()
        for event in self.watcher.update():
            self.on_change(event)

    def teardown(self) -> None:
        """Teardown the provider."""
        for provider in self.text_handlers:
            provider.teardown()
