"""Search orchestrator."""

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any, Self

import numpy as np
from more_itertools import flatten

from ..common.pydantic import SearchQuery, SearchResult
from ..events.watcher import ChangeType, FileChanged
from ..index.fs_index import FSIndex, PathMeta
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
        self._indexing_tasks: list[Future] = []
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def indexing_progress(self) -> tuple[int, int]:
        """Indexing progress."""
        total = len(self._indexing_tasks)
        finished = sum(1 for task in self._indexing_tasks if task.done())
        return total, finished

    @property
    def indexed_paths(self) -> list[Path]:
        """Indexed paths."""
        return self.fs_watcher.indexed_directories

    def __enter__(self) -> Self:
        """Start the search provider."""
        self.search_providers = [self._stack.enter_context(provider) for provider in self.search_providers]
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Stop the search provider."""
        self.fs_watcher.save()
        self._stack.close()

    def _handle_change(self, meta: PathMeta, removed: bool) -> None:
        """Handle a change."""
        event = FileChanged(path=meta.path, event_type=ChangeType.REMOVE if removed else ChangeType.UPSERT)
        for provider in self.search_providers:
            provider.update_index(event)
        if removed:
            self.fs_watcher.remove(meta)
        else:
            self.fs_watcher.add(meta)

    def update_selected_paths(self, directories: list[Path]) -> None:
        """Update the selected paths."""
        for task in self._indexing_tasks:
            task.cancel()
        wait(self._indexing_tasks)
        upserted, removed = self.fs_watcher.compute_diff(directories)
        remove_tasks = [self._executor.submit(self._handle_change, meta, True) for meta in removed]
        upsert_tasks = [self._executor.submit(self._handle_change, meta, False) for meta in upserted]
        self._indexing_tasks = upsert_tasks + remove_tasks

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results_nested = [provider.search(query) for provider in self.search_providers]
        results = list(flatten(results_nested))
        return normalize_results(results)
