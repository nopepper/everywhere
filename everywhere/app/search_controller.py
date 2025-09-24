"""Search orchestrator."""

from itertools import groupby
from pathlib import Path

import numpy as np
from more_itertools import flatten

from everywhere.events import add_callback

from ..common.pydantic import SearchQuery, SearchResult
from ..events.watcher import FileChanged
from ..search.search_provider import SearchProviderService
from ..watchers.fs_watcher import FSWatcher


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores."""
    if not scores:
        return []

    scores_np = np.array(scores)
    mu = scores_np.mean()
    normalized = (scores_np - mu) / (scores_np.max() - mu + 1e-8)
    normalized = np.clip(normalized, 0, 1)
    return normalized.tolist()


def normalize_results(results: list[SearchResult]) -> list[SearchResult]:
    """Normalize results."""
    results_agg: list[SearchResult] = []
    for _, group in groupby(sorted(results, key=lambda x: x.value), key=lambda x: x.value):
        results_agg.append(max(group, key=lambda x: x.confidence))

    new_scores = normalize_scores([x.confidence for x in results_agg])
    results_agg = [SearchResult(value=x.value, confidence=new_scores[i]) for i, x in enumerate(results_agg)]
    results_agg.sort(key=lambda x: x.confidence, reverse=True)
    return results_agg


class SearchController:
    """Search orchestrator service."""

    def __init__(self, search_providers: list[SearchProviderService], fs_watcher: FSWatcher):
        """Initialize the search controller."""
        self.search_providers = search_providers
        self.fs_watcher = fs_watcher

    @property
    def indexed_paths(self) -> list[Path]:
        """Indexed paths."""
        return self.fs_watcher.fs_path

    def start(self) -> None:
        """Start the search provider."""
        for provider in self.search_providers:
            provider.start()
        add_callback(FileChanged, self._handle_file_change)
        self.fs_watcher.start()

    def stop(self) -> None:
        """Stop the search provider."""
        for provider in self.search_providers:
            provider.stop()

    def update_selected_paths(self, paths: list[Path]) -> None:
        """Update the selected paths."""
        self.fs_watcher._restart_with_new_paths(paths)

    def _handle_file_change(self, event: FileChanged) -> None:
        """Handle a file changed event."""
        for provider in self.search_providers:
            provider.handle_file_change(event)

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results = list(flatten([provider.search(query) for provider in self.search_providers]))
        return normalize_results(results)
