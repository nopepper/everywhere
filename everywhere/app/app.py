"""Application entry point."""

from functools import partial
from pathlib import Path
from typing import Any, ClassVar

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Header
from textual.worker import WorkerCancelled

from everywhere.app.app_config import AppConfig
from everywhere.common.pydantic import SearchQuery, SearchResult
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


class _SearchCompleted(Message):
    def __init__(self, results: list[SearchResult]) -> None:
        super().__init__()
        self.results = results


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

    def _update_status_bar(self) -> None:
        status_bar = self.query_one(StatusBar)
        total, finished = self.controller.indexing_progress
        status_bar.total_tasks = total
        status_bar.finished_tasks = finished
        self.call_next(status_bar.refresh_progress)

    async def on_mount(self) -> None:
        """Set up the app when mounted."""
        self.selected_directories = self._config.selected_directories
        if len(self.selected_directories) == 0:
            self.notify("Please select directories to index")
            self.action_select_directories()
        self.indexing_timer = self.set_interval(0.2, self._update_status_bar, pause=False)

    @work(thread=True, exclusive=True)
    async def _launch_search(self, query: str) -> None:
        results = self.controller.search(SearchQuery(text=query))
        self.post_message(_SearchCompleted(results))

    @on(_SearchCompleted)
    def _on_search_completed(self, event: _SearchCompleted) -> None:
        with self.batch_update():
            self.query_one(ResultsTable).update_results(event.results)

    def on_search_bar_search_triggered(self, message: SearchBar.SearchTriggered) -> None:
        """Handle search requests from the search bar."""
        query = message.query
        self._search_debounced.submit(lambda: self.call_from_thread(self._launch_search, query))

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

        try:
            selected = await self.push_screen_wait(DirectorySelector(initial_paths=current_paths))
        except WorkerCancelled:
            return

        if not selected and len(current_paths) == 0:
            self.action_select_directories()
        elif selected:
            self.selected_directories = list(selected)

    def watch_selected_directories(self, old: list[Path], new: list[Path]) -> None:
        """Update the selected directories."""
        if old == new or len(new) == 0:
            return
        # Cancel any ongoing update before starting a new one
        self.controller.cancel_update()
        self.run_worker(
            partial(self.controller.update_selected_paths, new),
            thread=True,
            exclusive=True,
        )

    def action_close_app(self) -> None:
        """Close the application."""
        self.exit()

    def dump_config(self) -> AppConfig:
        """Dump the app config."""
        return AppConfig(
            embedding_search=self._config.embedding_search,
            selected_directories=self.selected_directories,
        )
