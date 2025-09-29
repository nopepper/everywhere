"""Search orchestrator."""

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from itertools import groupby
from pathlib import Path
from typing import Any, Self

import numpy as np
from more_itertools import flatten

from ..common.pydantic import SearchQuery, SearchResult
from ..index.document_index import DocumentIndex, IndexedDocument
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

    def __init__(self, search_providers: list[SearchProvider], doc_index: DocumentIndex):
        """Initialize the search controller."""
        self.search_providers = search_providers
        self.doc_index = doc_index
        self._stack = ExitStack()
        self._indexing_tasks: list[Future] = []
        self._executor = ThreadPoolExecutor(max_workers=1)

        for provider in search_providers:
            doc_index.register_provider(provider.provider_id)

    @property
    def indexing_progress(self) -> tuple[int, int]:
        """Indexing progress."""
        total = len(self._indexing_tasks)
        finished = sum(1 for task in self._indexing_tasks if task.done())
        return total, finished

    @property
    def indexed_paths(self) -> list[Path]:
        """Indexed paths."""
        return self.doc_index.indexed_directories

    def __enter__(self) -> Self:
        """Start the search provider."""
        self.search_providers = [self._stack.enter_context(provider) for provider in self.search_providers]
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Stop the search provider."""
        self.doc_index.save()
        self._executor.shutdown(wait=False)
        self._stack.close()

    def _handle_document(self, doc: IndexedDocument, removed: bool) -> None:
        """Handle document indexing or removal."""
        if removed:
            self.doc_index.remove(doc.path)
            return

        for provider in self.search_providers:
            if not doc.is_indexed_by(provider.provider_id):
                success = provider.index_document(doc)
                if success:
                    doc.mark_indexed_by(provider.provider_id)

    def update_selected_paths(self, directories: list[Path]) -> None:
        """Update the selected paths."""
        for task in self._indexing_tasks:
            task.cancel()
        wait(self._indexing_tasks)
        upserted, removed = self.doc_index.compute_diff(directories)
        remove_tasks = [self._executor.submit(self._handle_document, doc, True) for doc in removed]
        upsert_tasks = [self._executor.submit(self._handle_document, doc, False) for doc in upserted]
        self._indexing_tasks = upsert_tasks + remove_tasks

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results_nested = [provider.search(query) for provider in self.search_providers]
        results = list(flatten(results_nested))
        return normalize_results(results)
