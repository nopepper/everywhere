"""Results collector."""

import threading
from itertools import groupby

import numpy as np

from ..common.pydantic import SearchResult
from ..events import add_callback
from ..events.app import UserSearched
from ..events.search_provder import GotSearchResult

NORMALIZE_CONFIDENCE = True


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores."""
    if not scores:
        return []

    scores_np = np.array(scores)
    mu = scores_np.mean()
    normalized = (scores_np - mu) / (scores_np.max() - mu + 1e-8)
    normalized = np.clip(normalized, 0, 1)
    return normalized.tolist()


class ResultsCollector:
    """Results collector."""

    def __init__(self):
        """Initialize the results collector."""
        self._current_query = ""
        self._current_results: list[GotSearchResult] = []
        self._lock = threading.Lock()
        self._last_state = 0
        self._state = 0
        add_callback(UserSearched, self.on_user_searched)
        add_callback(GotSearchResult, self.on_got_search_result)

    def on_user_searched(self, event: UserSearched) -> None:
        """Handle user searched event."""
        with self._lock:
            self._current_query = event.query
            self._current_results = [r for r in self._current_results if r.query == event.query]
            self._state += 1

    def on_got_search_result(self, event: GotSearchResult) -> None:
        """Handle got search result event."""
        with self._lock:
            if event.query == self._current_query:
                self._current_results.append(event)
                self._state += 1

    @property
    def current_query(self) -> str:
        """Current query."""
        with self._lock:
            return self._current_query

    @property
    def has_new_results(self) -> bool:
        """Has new results."""
        return self._state > self._last_state

    def sync_results(self) -> tuple[str, list[SearchResult]]:
        """Current results."""
        with self._lock:
            self._last_state = self._state
            query = self._current_query
            results = [e.result for e in self._current_results]
        results_agg: list[SearchResult] = []
        for _, group in groupby(sorted(results, key=lambda x: x.value), key=lambda x: x.value):
            results_agg.append(max(group, key=lambda x: x.confidence))

        new_scores = normalize_scores([x.confidence for x in results_agg])
        results_agg = [SearchResult(value=x.value, confidence=new_scores[i]) for i, x in enumerate(results_agg)]
        results_agg.sort(key=lambda x: x.confidence, reverse=True)
        return query, results_agg
