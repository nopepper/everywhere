"""Progress tracker."""

from ..events import add_callback
from ..events.search_provder import IndexingFinished, IndexingStarted


class ProgressTracker:
    """Progress tracker."""

    def __init__(self):
        """Initialize the progress tracker."""
        self._total_tasks = 0
        self._finished_tasks = 0
        add_callback(str(id(self)), IndexingStarted, self.on_indexing_started)
        add_callback(str(id(self)), IndexingFinished, self.on_indexing_finished)

    def reset(self) -> None:
        """Reset the progress tracker."""
        self._total_tasks = 0
        self._finished_tasks = 0

    def on_indexing_started(self, event: IndexingStarted) -> None:
        """Handle indexing started event."""
        self._total_tasks += 1

    def on_indexing_finished(self, event: IndexingFinished) -> None:
        """Handle indexing finished event."""
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
        return self.total_tasks - self.finished_tasks == 0
