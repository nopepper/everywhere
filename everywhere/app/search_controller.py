"""Search orchestrator."""

from collections.abc import Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from threading import Event
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
    """Normalize and deduplicate search results."""
    # Deduplicate by path, keeping the highest confidence score
    results_agg: list[SearchResult] = []
    for _, group in groupby(sorted(results, key=lambda x: x.value), key=lambda x: x.value):
        results_agg.append(max(group, key=lambda x: x.confidence))

    # Normalize scores
    new_scores = normalize_scores([x.confidence for x in results_agg])

    # Create results with normalized scores
    normalized: list[SearchResult] = []
    for i, result in enumerate(results_agg):
        normalized.append(
            SearchResult(
                value=result.value,
                confidence=new_scores[i],
                size_bytes=result.size_bytes,
                last_modified_ns=result.last_modified_ns,
            )
        )

    # Sort by confidence descending
    normalized.sort(key=lambda x: x.confidence, reverse=True)
    return normalized


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
        self._cancel_event = Event()

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
        self._executor.shutdown(wait=False, cancel_futures=True)
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

    def cancel_update(self) -> None:
        """Cancel an ongoing update_selected_paths operation."""
        self._cancel_event.set()

    def update_selected_paths(self, directories: list[Path]) -> None:
        """Update the selected paths by walking directories and reconciling with index.

        Supports cancellation via cancel_update() method.
        """
        # Clear any previous cancellation and set up new one
        self._cancel_event.clear()

        def check_cancelled() -> None:
            """Check if cancellation has been requested."""
            if self._cancel_event.is_set():
                raise CancelledError("Update cancelled")

        # Cancel existing tasks and keep the ones that are not done
        self._indexing_tasks = [t for t in self._indexing_tasks if not t.cancel()]
        check_cancelled()

        # Get all paths currently in the database
        indexed_paths = self.doc_index.get_all_paths()
        check_cancelled()

        # Walk filesystem to get current state
        filesystem_paths: set[Path] = set()

        for path in walk_all_files(directories, self.path_filter):
            check_cancelled()
            try:
                stat = path.stat()
                filesystem_paths.add(path)
                current_mtime = stat.st_mtime
                current_size = stat.st_size

                # Get existing index entries for this path
                existing_rows = self.doc_index.get_rows_for_path(path)

                # Check for stale entries (different metadata) and schedule removals
                for last_modified, size, provider_id in existing_rows:
                    check_cancelled()
                    if last_modified != current_mtime or size != current_size:
                        # Stale entry - schedule removal
                        self._indexing_tasks.append(self._executor.submit(self._remove_document, path, provider_id))

                # Check if each provider needs to index this file
                for provider in self.search_providers:
                    check_cancelled()
                    if not self.doc_index.has_entry(path, current_mtime, current_size, provider.provider_id):
                        # Not indexed or stale - schedule indexing
                        doc = IndexedDocument(
                            path=path,
                            last_modified=current_mtime,
                            size=current_size,
                        )
                        self._indexing_tasks.append(
                            self._executor.submit(self._index_document, doc, provider.provider_id)
                        )

            except (OSError, FileNotFoundError):
                continue

        check_cancelled()

        # Remove documents that no longer exist in filesystem
        for path in indexed_paths:
            check_cancelled()
            if path not in filesystem_paths:
                # Get all providers that have this path indexed
                existing_rows = self.doc_index.get_rows_for_path(path)
                for _, _, provider_id in existing_rows:
                    check_cancelled()
                    self._indexing_tasks.append(self._executor.submit(self._remove_document, path, provider_id))

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results_nested = [provider.search(query) for provider in self.search_providers]
        results_nested = [normalize_results(list(results)) for results in results_nested]
        results = list(flatten(results_nested))
        # Normalize once individually, then together
        results = normalize_results(results)

        # Hydrate results with metadata from the index
        hydrated_results: list[SearchResult] = []
        for result in results:
            metadata = self.doc_index.get_metadata(result.value)
            if metadata is not None:
                size_bytes, last_modified_ns = metadata
                hydrated_results.append(
                    SearchResult(
                        value=result.value,
                        confidence=result.confidence,
                        size_bytes=size_bytes,
                        last_modified_ns=last_modified_ns,
                    )
                )
            else:
                hydrated_results.append(result)

        return hydrated_results
