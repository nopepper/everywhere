"""Status bar widget for the Everywhere app."""

from typing import Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Label, ProgressBar

from ..progress import ProgressTracker


class StatusBar(Horizontal):
    """A thin bottom status bar that shows indexing progress and status text.

    Polls the provided ProgressTracker every 100ms, shows a progress bar only
    when there are unfinished tasks, and always shows status text aligned to
    the bottom-right.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the status bar."""
        super().__init__(*args, **kwargs)
        self._progress_tracker = ProgressTracker()
        self._progress_bar = ProgressBar(id="status_progress_bar")
        self._warning_text = Label("Results may be incomplete", id="warning_text")
        self._spacer = Container(id="status_spacer")
        self._status_text = Label("", id="status_text")

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield self._progress_bar
        yield self._warning_text
        yield self._spacer
        yield self._status_text

    def on_mount(self) -> None:
        """Set up the status bar when mounted."""
        self.set_interval(0.1, self._refresh)
        self._progress_bar.total = 1
        self._progress_bar.progress = 0
        self._progress_bar.display = False
        self._warning_text.display = False
        self._spacer.display = True

    def _refresh(self) -> None:
        total, finished = self._progress_tracker.get_progress()
        remaining = max(0, total - finished)

        self._status_text.update(self._progress_tracker.status_text)

        # Show warning text when progress tracker is not ready
        if not self._progress_tracker.ready:
            if not self._warning_text.display:
                self._warning_text.display = True
        else:
            if self._warning_text.display:
                self._warning_text.display = False

        if remaining > 0 and total > 0:
            if not self._progress_bar.display:
                self._progress_bar.display = True
                self._spacer.display = False
            self._progress_bar.total = total
            self._progress_bar.progress = min(max(finished, 0), total)
        else:
            if self._progress_bar.display:
                self._progress_bar.display = False
                self._spacer.display = True
                self._progress_tracker.reset()
