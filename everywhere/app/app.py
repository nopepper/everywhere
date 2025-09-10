"""Application entry point."""

import threading
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.coordinate import Coordinate
from textual.reactive import reactive
from textual.widgets import DataTable, Header, Input

from everywhere.events.search_provder import GotSearchResult, IndexingFinished

from ..events import add_callback, publish
from ..events.app import UserSearched
from ..search_providers.onnx_text_search import ONNXTextSearchProvider
from ..watchers.fs_watcher import FSWatcher
from .collector import ResultsCollector
from .directory_selector import DirectorySelector
from .progress import ProgressTracker
from .status_bar import StatusBar

DEBOUNCE_LATENCY = 0.1
RESULT_LIMIT = 1000


def format_size(size_bytes: int) -> str:
    """Format file size in bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB"]
    unit_index = 0
    size = float(size_bytes)

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def format_date(timestamp_ns: int) -> str:
    """Format nanosecond timestamp to human readable date."""
    timestamp_s = timestamp_ns / 1_000_000_000
    dt = datetime.fromtimestamp(timestamp_s)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def confidence_to_color(confidence: float) -> Text:
    """Convert confidence (0-1) into a gray→green gradient cell."""
    confidence = max(0.0, min(1.0, confidence**2))

    # Start at light gray (#d0d0d0) → End at bright green (#00ff00)
    start_r, start_g, start_b = 208, 208, 208
    end_r, end_g, end_b = 0, 255, 0

    r = round(start_r + (end_r - start_r) * confidence)
    g = round(start_g + (end_g - start_g) * confidence)
    b = round(start_b + (end_b - start_b) * confidence)

    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    return Text("  ", style=f"on {hex_color}")


class DirectoryIndexCommand(Provider):
    """Command provider for directory indexing and application commands."""

    async def discover(self) -> Hits:
        """Expose common actions in the command palette."""
        app = self.app
        if isinstance(app, EverywhereApp):
            yield DiscoveryHit(
                "Pick indexed directories...", app.action_select_directories, help="Select directories to be indexed"
            )
            yield DiscoveryHit("Close application", app.action_close_app, help="Close the application")

    async def search(self, query: str) -> Hits:
        """Return actions when the query matches."""
        app = self.app
        if not isinstance(app, EverywhereApp):
            return

        matcher = self.matcher(query)

        # Directory selection command
        if query.lower().startswith("pick") or query.lower().startswith("dir") or query.lower().startswith("index"):
            yield DiscoveryHit(
                "Pick indexed directories...",
                app.action_select_directories,
                help="Select directories to be indexed",
            )

        # Close application command
        close_command = "Close application"
        score = matcher.match(close_command)
        if score > 0:
            yield Hit(
                score,
                matcher.highlight(close_command),
                app.action_close_app,
                help="Close the application",
            )


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

    def __init__(self, fs_path: str = "data_test"):
        """Initialize the Everything app.

        Args:
            fs_path: Path to the directory to search in.
        """
        super().__init__()

        search_providers = [
            ONNXTextSearchProvider(
                onnx_model_path=Path("models/all-MiniLM-L6-v2/onnx/model_quint8_avx2.onnx"),
                tokenizer_path=Path("models/all-MiniLM-L6-v2"),
            )
        ]
        self.search_providers = [provider.start_eventful() for provider in search_providers]
        self.watcher = FSWatcher(
            fs_path=Path(fs_path),
            supported_types=set.union(*[provider.supported_types for provider in search_providers]),
        )
        self.watcher.start()
        self._progress = ProgressTracker()
        self._results_collector = ResultsCollector()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Input(placeholder="Search files and folders...", classes="search-input", id="search_input")
        yield DataTable(id="results_table")
        yield StatusBar(progress=self._progress)

    def on_mount(self) -> None:
        """Set up the app when mounted."""
        table = self.query_one("#results_table", DataTable)
        table.add_columns("", "Name", "Path", "Size", "Date Modified")
        table.cursor_type = "cell"
        self._update_table_timer = None
        add_callback(IndexingFinished, self.on_indexing_finished)
        add_callback(GotSearchResult, self.on_got_search_result)

    def on_indexing_finished(self, event: IndexingFinished) -> None:
        """Handle indexing finished event."""
        self._publish_search_soon(self.query_one("#search_input", Input).value)

    def on_got_search_result(self, event: GotSearchResult) -> None:
        """Handle got search result event."""
        self._update_table_soon()

    def on_show(self) -> None:
        """Called when the widget becomes visible - layout is ready."""
        self._fit_table_columns()

    def _fit_table_columns(self) -> None:
        """Fit DataTable columns to widget width (Confidence=2, Size=5, Date=5; Name:Path = 1:3)."""
        table = self.query_one("#results_table", DataTable)

        console_width = self.console.size.width if hasattr(self.console, "size") else 0

        estimated_table_width = console_width - 2  # 2 for the margins

        total = estimated_table_width

        if total <= 10:  # Need some reasonable minimum
            return

        if len(table.ordered_columns) != 5:
            return

        CONF, SIZE, DATE = 2, 8, 20  # noqa: N806
        OVERHEAD = 12  # noqa: N806

        fixed = CONF + SIZE + DATE + OVERHEAD
        free = max(0, total - fixed)

        # 1:3 split
        name_w = max(25, free // 4)
        path_w = max(1, free - name_w)

        widths = [CONF, name_w, path_w, SIZE, DATE]
        for index, desired in enumerate(widths):
            column = table.ordered_columns[index]
            column.auto_width = False
            column.width = desired

        if hasattr(table, "_require_update_dimensions"):
            table._require_update_dimensions = True  # type: ignore[attr-defined]
        table.refresh()

    def on_resize(self, _: Any) -> None:
        """Handle terminal resize."""
        self._fit_table_columns()

    def _publish_search_soon(self, search_term: str) -> None:
        """Publish a search soon (debounced)."""
        # Reject empty searches
        if search_term == "":
            self._update_table_soon()
            return

        if hasattr(self, "_search_timer") and self._search_timer is not None:
            self._search_timer.cancel()

        def _publish_search():
            publish(UserSearched(query=search_term))
            self._search_timer = None

        self._search_timer = threading.Timer(DEBOUNCE_LATENCY, _publish_search)
        self._search_timer.start()

    def _update_table_soon(self) -> None:
        """Update table soon (debounced)."""
        if hasattr(self, "_update_table_timer") and self._update_table_timer is not None:
            self._update_table_timer.cancel()

        def _update_table():
            self._update_results_table()
            self._update_table_timer = None

        self._update_table_timer = threading.Timer(DEBOUNCE_LATENCY, _update_table)
        self._update_table_timer.start()

    def on_input_changed(self, message: Input.Changed) -> None:
        """Handle search input changes."""
        if message.input.id != "search_input":
            return
        self._publish_search_soon(message.value)

    def _update_results_table(self) -> None:
        current_search_term = self.query_one("#search_input", Input).value
        table = self.query_one("#results_table", DataTable)

        if current_search_term.strip() == "" and table.row_count > 0:
            table.clear()
            return

        if not self._results_collector.has_new_results:
            return

        current_query, results = self._results_collector.sync_results()
        if current_search_term != current_query:
            return

        with self.batch_update():
            for i, result in enumerate(results):
                path = result.value
                if not path.exists():
                    continue
                confidence_label = confidence_to_color(result.confidence)
                stat = path.stat()
                new_col_values = [
                    confidence_label,
                    path.name,
                    str(path),
                    format_size(stat.st_size),
                    format_date(stat.st_mtime_ns),
                ]

                if i < table.row_count:
                    old_col_values = table.get_row_at(i)
                    for j, (old_value, new_value) in enumerate(zip(old_col_values, new_col_values, strict=True)):
                        if old_value != new_value:
                            table.update_cell_at(Coordinate(row=i, column=j), new_value)
                else:
                    table.add_row(*new_col_values)

            if table.row_count > len(results):
                keys_to_remove = [
                    table.coordinate_to_cell_key(Coordinate(row=row_index, column=0))[0]
                    for row_index in range(len(results), table.row_count)
                ]
                for row_key in keys_to_remove:
                    table.remove_row(row_key)

    @work
    async def action_select_directories(self) -> None:
        """Open the directory selection dialog."""
        current_paths = self.watcher.fs_path
        if isinstance(current_paths, Path):
            current_paths = [current_paths]

        worker = self.run_worker(self.push_screen_wait(DirectorySelector(initial_paths=current_paths)))
        selected = await worker.wait()  # returns the dismissal value

        if not selected:
            return

        old_watcher_dump = self.watcher.model_dump()
        old_watcher_dump["fs_path"] = list(selected)
        self.watcher.stop()
        del self.watcher
        self.watcher = FSWatcher(**old_watcher_dump)
        self.watcher.start()

    def action_close_app(self) -> None:
        """Close the application."""
        self.exit()


def main():
    """Main entry point."""
    app = EverywhereApp()
    app.run()


if __name__ == "__main__":
    main()
