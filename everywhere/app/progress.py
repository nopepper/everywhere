"""Progress tracker."""

import threading

from ..events import add_callback
from ..events.ann import IndexSaveFinished, IndexSaveStarted
from ..events.search_provder import GotIndexingRequest, IndexingFinished


class ProgressTracker:
    """Progress tracker."""

    def __init__(self):
        """Initialize the progress tracker."""
        self._total_tasks = 0
        self._finished_tasks = 0
        self._lock = threading.Lock()
        add_callback(GotIndexingRequest, self.on_indexing_request)
        add_callback(IndexingFinished, self.on_indexing_finished)
        add_callback(IndexSaveStarted, self.on_index_save_started)
        add_callback(IndexSaveFinished, self.on_index_save_finished)

    def reset(self) -> None:
        """Reset the progress tracker."""
        with self._lock:
            self._total_tasks = 0
            self._finished_tasks = 0

    def on_index_save_started(self, event: IndexSaveStarted) -> None:
        """Handle index save started event."""
        with self._lock:
            self._total_tasks += 1

    def on_index_save_finished(self, event: IndexSaveFinished) -> None:
        """Handle index save finished event."""
        with self._lock:
            self._finished_tasks += 1

    def on_indexing_request(self, event: GotIndexingRequest) -> None:
        """Handle indexing started event."""
        with self._lock:
            self._total_tasks += 1

    def on_indexing_finished(self, event: IndexingFinished) -> None:
        """Handle indexing finished event."""
        with self._lock:
            self._finished_tasks += 1

    @property
    def status_text(self) -> str:
        """Status text."""
        if self.total_tasks - self.finished_tasks == 0:
            return "Ready"
        return f"Indexing {self.finished_tasks}/{self.total_tasks}"

    @property
    def total_tasks(self) -> int:
        """Total tasks."""
        return self._total_tasks

    @property
    def finished_tasks(self) -> int:
        """Open tasks."""
        return self._finished_tasks

    @property
    def ready(self) -> bool:
        """Ready."""
        return self.total_tasks - self.finished_tasks <= 0
