"""Results table widget."""

from datetime import datetime
from itertools import groupby
from typing import Any

import numpy as np
from rich.text import Text
from textual.reactive import reactive
from textual.widgets import DataTable

from ...common.debounce import DebouncedRunner
from ...common.pydantic import SearchResult
from ...events import add_callback
from ...events.app import AppResized, UserSearched
from ...events.search_provder import GotSearchResult


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores."""
    if not scores:
        return []

    scores_np = np.array(scores)
    mu = scores_np.mean()
    normalized = (scores_np - mu) / (scores_np.max() - mu + 1e-8)
    normalized = np.clip(normalized, 0, 1)
    return normalized.tolist()


def _format_size(n: int) -> str:
    if n == 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i, s = 0, float(n)
    while s >= 1024 and i < len(units) - 1:
        s /= 1024
        i += 1
    if i == 0:
        return f"{int(s)} {units[i]}"
    else:
        return f"{s:.1f} {units[i]}"


def _format_date(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1_000_000_000).strftime("%Y-%m-%d %H:%M:%S")


def _confidence_chip(p: float) -> Text:
    p = max(0.0, min(1.0, p * p))
    r = round(208 + (0 - 208) * p)
    g = round(208 + (255 - 208) * p)
    b = round(208 + (0 - 208) * p)
    return Text("  ", style=f"on #{r:02x}{g:02x}{b:02x}")


DEBOUNCE_LATENCY = 0.1
RESULT_LIMIT = 1000


class ResultsTable(DataTable):
    """Owns sizing and diff-updates."""

    search_query: reactive[str] = reactive("")
    search_results: reactive[list[SearchResult]] = reactive(list)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the results table."""
        super().__init__(*args, **kwargs)
        self._update_table_debounced = DebouncedRunner(DEBOUNCE_LATENCY)

    def on_mount(self) -> None:
        """Set up the table when mounted."""
        self.add_columns("", "Name", "Path", "Size", "Date Modified")
        self.cursor_type = "cell"
        add_callback(AppResized, self._on_size_changed)
        add_callback(GotSearchResult, self._on_got_search_result)
        add_callback(UserSearched, self._on_user_searched)

    def _on_user_searched(self, event: UserSearched) -> None:
        """Handle user searched event."""
        self.search_query = event.query

    def watch_search_query(self, old_query: str, new_query: str) -> None:
        """Handle search query changes."""
        if old_query != new_query:
            self.search_results = []

    def _on_got_search_result(self, event: GotSearchResult) -> None:
        """Handle got search result event."""
        self.search_results = [*self.search_results, event.result]

    def _on_size_changed(self, msg: AppResized) -> None:
        """Handle size changes."""
        if len(self.ordered_columns) != 5:
            return
        console_width = msg.width
        total = max(0, console_width - 2)  # margins
        if total <= 10:
            return
        CONF, SIZE, DATE, OVERHEAD = 2, 8, 20, 12  # noqa: N806
        fixed = CONF + SIZE + DATE + OVERHEAD
        free = max(0, total - fixed)
        name_w = max(25, free // 4)
        path_w = max(1, free - name_w)
        widths = [CONF, name_w, path_w, SIZE, DATE]
        for i, w in enumerate(widths):
            col = self.ordered_columns[i]
            col.auto_width = False
            col.width = w
        # internal refresh hint
        if hasattr(self, "_require_update_dimensions"):
            self._require_update_dimensions = True
        self.refresh()

    def watch_search_results(self, _: Any, results: list[SearchResult]) -> None:
        """Handle search results."""
        self._update_table_debounced.submit(lambda: self._redraw_results(results))

    def _normalize_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Normalize results."""
        results_agg: list[SearchResult] = []
        for _, group in groupby(sorted(results, key=lambda x: x.value), key=lambda x: x.value):
            results_agg.append(max(group, key=lambda x: x.confidence))

        new_scores = normalize_scores([x.confidence for x in results_agg])
        results_agg = [SearchResult(value=x.value, confidence=new_scores[i]) for i, x in enumerate(results_agg)]
        results_agg.sort(key=lambda x: x.confidence, reverse=True)
        return results_agg

    def _redraw_results(self, results: list[SearchResult]) -> None:
        """Handle search results."""
        self.clear()  # wipes all rows
        for result in self._normalize_results(results):
            path = result.value
            try:
                if not path.exists():
                    continue
                stat = path.stat()
                self.add_row(
                    _confidence_chip(result.confidence),
                    path.name,
                    str(path),
                    _format_size(stat.st_size),
                    _format_date(stat.st_mtime_ns),
                )
            except Exception as e:
                self.notify(f"Error adding result: {e}")
