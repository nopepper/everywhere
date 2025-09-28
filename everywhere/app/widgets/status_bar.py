"""Status bar widget for the Everywhere app."""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Label, ProgressBar

from ...common.debounce import DebouncedRunner
from ...events import add_callback
from ...events.ann import IndexSaveFinished, IndexSaveStarted
from ...events.search_provider import GotIndexingRequest, IndexingFinished

DEBOUNCE_LATENCY = 0.1


class StatusBar(Horizontal):
    """A thin bottom status bar that shows indexing progress and status text.

    Polls the provided ProgressTracker every 100ms, shows a progress bar only
    when there are unfinished tasks, and always shows status text aligned to
    the bottom-right.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the status bar."""
        super().__init__(*args, **kwargs)
        self.progress_bar = ProgressBar(id="status_progress_bar")
        self.warning_text = Label("Results may be incomplete", id="warning_text")
        self.spacer = Container(id="status_spacer")
        self.status_text = Label("", id="status_text")
        self.total_tasks = 0
        self.finished_tasks = 0
        self._debounced = DebouncedRunner(DEBOUNCE_LATENCY)

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield self.progress_bar
        yield self.warning_text
        yield self.spacer
        yield self.status_text

    def on_mount(self) -> None:
        """Set up the status bar when mounted."""
        add_callback(GotIndexingRequest, self._increment_tasks)
        add_callback(IndexSaveStarted, self._increment_tasks)
        add_callback(IndexingFinished, self._decrement_tasks)
        add_callback(IndexSaveFinished, self._decrement_tasks)

    def set_progress_visibility(self, display: bool) -> None:
        """Set the progress bar visibility."""
        self.progress_bar.display = display
        self.warning_text.display = display
        self.spacer.display = not display

    def _increment_tasks(self, event: GotIndexingRequest | IndexSaveStarted) -> None:
        """Increment tasks."""
        self.total_tasks += 1
        self._debounced.submit(self.refresh_progress)

    def _decrement_tasks(self, event: IndexingFinished | IndexSaveFinished) -> None:
        """Decrement tasks."""
        self.finished_tasks += 1
        self._debounced.submit(self.refresh_progress)

    def refresh_progress(self) -> None:
        """Refresh the status bar."""
        self.progress_bar.total = self.total_tasks
        self.progress_bar.progress = self.finished_tasks
        if self.finished_tasks >= self.total_tasks:
            self.set_progress_visibility(False)
            self.status_text.update("Ready")
        else:
            self.set_progress_visibility(True)
            self.status_text.update(f"Indexing {self.finished_tasks}/{self.total_tasks}")
