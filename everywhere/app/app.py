"""Application entry point."""

from pathlib import Path
from typing import Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Header

from everywhere.common.pydantic import SearchQuery

from ..common.debounce import DebouncedRunner
from ..events import add_callback, publish
from ..events.app import AppResized, UserSearched, UserSelectedDirectories
from .app_config import get_app_components
from .commands.directory_index import DirectoryIndexCommand
from .screens.directory_selector import DirectorySelector
from .widgets.results_table import ResultsTable
from .widgets.search_bar import SearchBar
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
    SearchBar {
        dock: top;
        height: 3;
        margin: 1;
    }

    SearchBar > Input {
        height: 1fr;
        width: 1fr;
    }

    DataTable {
        height: 1fr;
        width: 1fr;
        margin: 0 1;
    }

    .search-input {
        border: solid $accent;
    }

    StatusBar {
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

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield SearchBar()
        yield ResultsTable()
        yield StatusBar()

    async def on_mount(self) -> None:
        """Set up the app when mounted."""
        # If no paths are configured, prompt the user to select some until they do
        if len(get_app_components().indexed_paths) == 0:
            self.notify("Please select directories to index")
            self.action_select_directories()
        self._search_debounced = DebouncedRunner(DEBOUNCE_LATENCY)
        add_callback(UserSearched, self.on_user_searched)

    def search_and_update(self, query: str) -> None:
        """Search and update the results table."""
        results = get_app_components().search(SearchQuery(text=query))
        self.query_one(ResultsTable).update_results(results)

    def on_user_searched(self, event: UserSearched) -> None:
        """Handle user searched event."""
        self._search_debounced.submit(lambda: self.search_and_update(event.query))

    def on_show(self) -> None:
        """Called when the widget becomes visible - layout is ready."""
        publish(AppResized(width=self.console.size.width, height=self.console.size.height))

    def on_resize(self, _: Any) -> None:
        """Handle terminal resize."""
        publish(AppResized(width=self.console.size.width, height=self.console.size.height))

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
