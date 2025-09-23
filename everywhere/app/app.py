"""Application entry point."""

from pathlib import Path
from typing import Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Header, Input

from ..common.debounce import DebouncedRunner
from ..events import add_callback, publish
from ..events.app import AppResized, UserSearched, UserSelectedDirectories
from ..events.search_provder import IndexingFinished
from .app_config import get_app_components
from .commands.directory_index import DirectoryIndexCommand
from .progress import ProgressTracker
from .screens.directory_selector import DirectorySelector
from .widgets.results_table import ResultsTable
from .widgets.status_bar import StatusBar

DEBOUNCE_LATENCY = 0.1


class EverywhereApp(App):
    """File search application."""

    TITLE = "Everywhere"
    COMMANDS: ClassVar = {DirectoryIndexCommand}
    BINDINGS: ClassVar = [
        Binding("ctrl+c", "close_app", "Close application", priority=True),
    ]

    CSS = """
    Input {
        dock: top;
        height: 3;
        margin: 1;
    }

    DataTable {
        height: 1fr;
        width: 1fr;
        margin: 0 1;
    }

    .search-input {
        border: solid $accent;
    }

    .status-bar {
        dock: bottom;
        height: 1;
        padding: 0 1;
    }

    #status_progress_bar {
        width: 1fr;
        margin-right: 1;
    }

    #status_spacer {
        width: 1fr;
    }

    #warning_text {
        width: auto;
        color: yellow;
        text-style: bold;
        margin-right: 1;
    }

    #status_text {
        width: 25;
        content-align: right middle;
        min-width: 25;
    }
    """

    search_term = reactive("")

    def __init__(self):
        """Initialize the Everything app."""
        super().__init__()
        self._progress = ProgressTracker()
        self._search_debounced = DebouncedRunner(DEBOUNCE_LATENCY)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Input(placeholder="Search files and folders...", classes="search-input", id="search_input")
        yield ResultsTable(id="results_table")
        yield StatusBar(progress=self._progress)

    async def on_mount(self) -> None:
        """Set up the app when mounted."""
        add_callback(IndexingFinished, self._on_indexing_finished)

        # If no paths are configured, prompt the user to select some until they do
        if len(get_app_components().indexed_paths) == 0:
            self.notify("Please select directories to index")
            self.action_select_directories()

    def _on_indexing_finished(self, _: Any) -> None:
        self._search_debounced.submit(lambda: publish(UserSearched(query=self.query_one("#search_input", Input).value)))

    def on_show(self) -> None:
        """Called when the widget becomes visible - layout is ready."""
        publish(AppResized(width=self.console.size.width, height=self.console.size.height))

    def on_resize(self, _: Any) -> None:
        """Handle terminal resize."""
        publish(AppResized(width=self.console.size.width, height=self.console.size.height))

    def on_input_changed(self, message: Input.Changed) -> None:
        """Handle search input changes."""
        if message.input.id != "search_input":
            return
        self._search_debounced.submit(lambda: publish(UserSearched(query=message.value)))

    @work
    async def action_select_directories(self) -> None:
        """Open the directory selection dialog."""
        current_paths = get_app_components().indexed_paths
        if isinstance(current_paths, Path):
            current_paths = [current_paths]

        worker = self.run_worker(self.push_screen_wait(DirectorySelector(initial_paths=current_paths)))
        selected = await worker.wait()  # returns the dismissal value

        if not selected and len(get_app_components().indexed_paths) == 0:
            self.action_select_directories()
            return

        if not selected:
            return

        publish(UserSelectedDirectories(directories=list(selected)))

    def action_close_app(self) -> None:
        """Close the application."""
        self.exit()
