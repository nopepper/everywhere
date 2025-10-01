"""Search orchestrator."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any, Self

import numpy as np
from more_itertools import flatten

from ..common.pydantic import SearchQuery, SearchResult
from ..index.document_index import DocumentIndex, IndexedDocument, walk_all_files
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

    def __init__(
        self,
        search_providers: list[SearchProvider],
        doc_index: DocumentIndex,
        path_filter: Callable[[Path], bool] | None = None,
    ):
        """Initialize the search controller."""
        self.search_providers = search_providers
        self.doc_index = doc_index
        self.path_filter = path_filter
        self._stack = ExitStack()
        self._indexing_tasks: list[Future] = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._providers_by_id = {provider.provider_id: provider for provider in search_providers}

    @property
    def indexing_progress(self) -> tuple[int, int]:
        """Indexing progress."""
        total = len(self._indexing_tasks)
        finished = sum(1 for task in self._indexing_tasks if task.done())
        return total, finished

    @property
    def indexed_paths(self) -> list[Path]:
        """Indexed paths."""
        return sorted(self.doc_index.get_all_paths())

    def __enter__(self) -> Self:
        """Start the search provider."""
        self.search_providers = [self._stack.enter_context(provider) for provider in self.search_providers]
        self._providers_by_id = {provider.provider_id: provider for provider in self.search_providers}
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Stop the search provider."""
        self.doc_index.save()
        self.doc_index.close()
        self._executor.shutdown(wait=True)
        self._stack.close()

    def _remove_document(self, path: Path, provider_id: str) -> None:
        """Remove a document from a provider and the index."""
        provider = self._providers_by_id.get(provider_id)
        if provider:
            success = provider.remove_document(path)
            if success:
                self.doc_index.remove(path, provider_id)

    def _index_document(self, doc: IndexedDocument, provider_id: str) -> None:
        """Index a document with a provider and add to index."""
        provider = self._providers_by_id.get(provider_id)
        if provider:
            success = provider.index_document(doc)
            if success:
                self.doc_index.add(doc.path, doc.last_modified, doc.size, provider_id)

    def update_selected_paths(self, directories: list[Path]) -> None:
        """Update the selected paths by walking directories and reconciling with index."""
        # Cancel existing tasks
        for task in self._indexing_tasks:
            task.cancel()
        wait(self._indexing_tasks)
        self._indexing_tasks = []

        # Get all paths currently in the database
        indexed_paths = self.doc_index.get_all_paths()

        # Walk filesystem to get current state
        filesystem_paths: set[Path] = set()
        tasks: list[Future] = []

        for path in walk_all_files(directories, self.path_filter):
            try:
                stat = path.stat()
                filesystem_paths.add(path)
                current_mtime = stat.st_mtime
                current_size = stat.st_size

                # Get existing index entries for this path
                existing_rows = self.doc_index.get_rows_for_path(path)

                # Check for stale entries (different metadata) and schedule removals
                for last_modified, size, provider_id in existing_rows:
                    if last_modified != current_mtime or size != current_size:
                        # Stale entry - schedule removal
                        tasks.append(self._executor.submit(self._remove_document, path, provider_id))

                # Check if each provider needs to index this file
                for provider in self.search_providers:
                    if not self.doc_index.has_entry(path, current_mtime, current_size, provider.provider_id):
                        # Not indexed or stale - schedule indexing
                        doc = IndexedDocument(
                            path=path,
                            last_modified=current_mtime,
                            size=current_size,
                        )
                        tasks.append(self._executor.submit(self._index_document, doc, provider.provider_id))

            except (OSError, FileNotFoundError):
                continue

        # Remove documents that no longer exist in filesystem
        for path in indexed_paths:
            if path not in filesystem_paths:
                # Get all providers that have this path indexed
                existing_rows = self.doc_index.get_rows_for_path(path)
                for _, _, provider_id in existing_rows:
                    tasks.append(self._executor.submit(self._remove_document, path, provider_id))

        self._indexing_tasks = tasks

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results_nested = [provider.search(query) for provider in self.search_providers]
        results = list(flatten(results_nested))
        return normalize_results(results)
