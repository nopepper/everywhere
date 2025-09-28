"""Application entry point."""

import threading
from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.reactive import reactive
from textual.widgets import Header

from everywhere.app.app_config import AppConfig
from everywhere.common.pydantic import SearchQuery
from everywhere.events import publish

from ..common.debounce import DebouncedRunner
from ..events.app import AppResized
from .commands.directory_index import DirectoryIndexCommand
from .screens.directory_selector import DirectorySelector
from .search_controller import SearchController
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

    selected_directories: reactive[list[Path]] = reactive([])

    def __init__(self, config: AppConfig, controller: SearchController):
        """Initialize the app."""
        super().__init__()
        self._config = config
        self.controller = controller
        self._search_debounced = DebouncedRunner(DEBOUNCE_LATENCY)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield SearchBar()
        yield ResultsTable()
        yield StatusBar()

    async def on_mount(self) -> None:
        """Set up the app when mounted."""
        self.selected_directories = self._config.selected_directories
        if len(self.selected_directories) == 0:
            self.notify("Please select directories to index")
            self.action_select_directories()

    def search_and_update(self, query: str) -> None:
        """Search and update the results table."""
        results = self.controller.search(SearchQuery(text=query))
        with self.batch_update():
            self.query_one(ResultsTable).update_results(results)

    def on_search_bar_search_triggered(self, message: SearchBar.SearchTriggered) -> None:
        """Handle search requests from the search bar."""
        self._search_debounced.submit(partial(self.search_and_update, message.query))

    def on_show(self) -> None:
        """Called when the widget becomes visible - layout is ready."""
        publish(AppResized(width=self.console.size.width, height=self.console.size.height))

    def on_resize(self, _: Any) -> None:
        """Handle terminal resize."""
        publish(AppResized(width=self.console.size.width, height=self.console.size.height))

    @work
    async def action_select_directories(self) -> None:
        """Open the directory selection dialog."""
        current_paths = self.selected_directories
        if isinstance(current_paths, Path):
            current_paths = [current_paths]

        worker = self.run_worker(self.push_screen_wait(DirectorySelector(initial_paths=current_paths)))
        selected = await worker.wait()  # returns the dismissal value
        if not selected and len(current_paths) == 0:
            self.action_select_directories()
        elif selected:
            self.selected_directories = list(selected)

    def watch_selected_directories(self, old: list[Path], new: list[Path]) -> None:
        """Update the selected directories."""
        threading.Thread(target=self.controller.update_selected_paths, args=(new,), daemon=True).start()

    def action_close_app(self) -> None:
        """Close the application."""
        self.exit()

    def dump_config(self) -> AppConfig:
        """Dump the app config."""
        return AppConfig(
            embedding_search=self._config.embedding_search,
            selected_directories=self.selected_directories,
        )
