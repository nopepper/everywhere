"""Search orchestrator."""

from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any, Self

import numpy as np
from more_itertools import flatten

from ..common.pydantic import SearchQuery, SearchResult
from ..events import publish
from ..events.search_provider import GotIndexingRequest, IndexingFinished, IndexingStarted
from ..index.fs_index import FSIndex
from ..search.search_provider import SearchProvider


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

    def __init__(self, search_providers: list[SearchProvider], fs_index: FSIndex):
        """Initialize the search controller."""
        self.search_providers = search_providers
        self.fs_watcher = fs_index
        self._stack = ExitStack()

    @property
    def indexed_paths(self) -> list[Path]:
        """Indexed paths."""
        return self.fs_watcher.indexed_directories

    def __enter__(self) -> Self:
        """Start the search provider."""
        self.search_providers = [self._stack.enter_context(provider) for provider in self.search_providers]
        self.update_selected_paths(self.indexed_paths)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Stop the search provider."""
        self.fs_watcher.save()
        self._stack.close()

    def update_selected_paths(self, directories: list[Path]) -> None:
        """Update the selected paths."""
        changes = []
        for change in self.fs_watcher.update_fs_paths(directories):
            publish(GotIndexingRequest(path=change.path))
            changes.append(change)

        for change in changes:
            for provider in self.search_providers:
                publish(IndexingStarted(path=change.path))
                provider.update_index(change)
                publish(IndexingFinished(path=change.path, success=True))

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results = list(flatten([provider.search(query) for provider in self.search_providers]))
        return normalize_results(results)
