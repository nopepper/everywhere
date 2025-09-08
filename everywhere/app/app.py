"""Application entry point."""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import DiscoveryHit, Hit, Hits, Provider
from textual.reactive import reactive
from textual.widgets import DataTable, Header, Input

from ..common.pydantic import SearchResult
from ..search_providers.fs_search import FSSearchProvider
from ..search_providers.onnx_text_search import ONNXTextSearchProvider
from ..search_providers.search_provider import SearchQuery
from ..watchers.fs_watcher import FSWatcher
from .directory_selector import DirectorySelector
from .progress import ProgressTracker
from .status_bar import StatusBar

DEBOUNCE_LATENCY = 0.05
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
    confidence = max(0.0, min(1.0, confidence))

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

        self.text_search_provider = ONNXTextSearchProvider(
            onnx_model_path=Path("models/all-MiniLM-L6-v2/onnx/model_quint8_avx2.onnx"),
            tokenizer_path=Path("models/all-MiniLM-L6-v2"),
        )
        watcher = FSWatcher(fs_path=Path(fs_path))
        self.fs_search = FSSearchProvider(search_providers=[self.text_search_provider], watcher=watcher)
        self.search_setup_done = False
        self._debounce_task: asyncio.Task | None = None
        self._search_gen: int = 0
        self._progress = ProgressTracker()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Input(placeholder="Search files and folders...", classes="search-input", id="search_input")
        yield DataTable(id="results_table")
        yield StatusBar(progress=self._progress)

    def on_mount(self) -> None:
        """Set up the app when mounted."""
        table = self.query_one("#results_table", DataTable)
        table.add_column("", width=2, key="confidence")
        table.add_column("Name", width=24, key="file_name")
        table.add_column("Path", width=100, key="path")
        table.add_column("Size", width=9, key="size")
        table.add_column("Date Modified", width=19, key="date_modified")
        table.cursor_type = "cell"
        self._setup_task = asyncio.create_task(self._setup_search())

    async def _setup_search(self) -> None:
        try:
            await asyncio.get_event_loop().run_in_executor(None, self.fs_search.setup)
            self.search_setup_done = True
            if self.search_term:
                await self._perform_search(self._search_gen)
        except Exception as e:
            self.notify(f"Failed to initialize search: {e}", severity="error")

    def on_input_changed(self, message: Input.Changed) -> None:
        """Handle search input changes."""
        if message.input.id != "search_input":
            return
        self.search_term = message.value
        self._search_gen += 1
        gen = self._search_gen

        if self._debounce_task and not self._debounce_task.done():
            self._debounce_task.cancel()

        self._debounce_task = asyncio.create_task(self._debounced_search(gen))

    async def _debounced_search(self, gen: int) -> None:
        try:
            await asyncio.sleep(DEBOUNCE_LATENCY)
        except asyncio.CancelledError:
            return
        await self._perform_search(gen)

    async def _perform_search(self, gen: int) -> None:
        """Perform search and update results if still the latest generation."""
        if not self.search_term.strip() or not self.search_setup_done:
            if gen == self._search_gen:
                table = self.query_one("#results_table", DataTable)
                table.clear()
            return

        try:
            search_query = SearchQuery(text=self.search_term)
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: list(self.fs_search.search(search_query))
            )
            results = results[:RESULT_LIMIT]

            if gen == self._search_gen:
                await self._update_results_table(results)
        except Exception as e:
            if gen == self._search_gen:
                self.notify(f"Search error: {e}", severity="error")

    async def _update_results_table(self, results: list[SearchResult]) -> None:
        table = self.query_one("#results_table", DataTable)
        table.clear()

        for result in results:
            path = result.value
            confidence_label = confidence_to_color(result.confidence)
            try:
                stat = path.stat()
                table.add_row(
                    confidence_label,
                    path.name,
                    str(path),
                    format_size(stat.st_size),
                    format_date(stat.st_mtime_ns),
                )
            except (OSError, FileNotFoundError):
                table.add_row(path.name, str(path), "N/A", "N/A", label=confidence_label)

    @work
    async def action_select_directories(self) -> None:
        """Open the directory selection dialog."""
        current_paths = self.fs_search.watcher.fs_path
        if isinstance(current_paths, Path):
            current_paths = [current_paths]

        worker = self.run_worker(self.push_screen_wait(DirectorySelector(initial_paths=current_paths)))
        selected = await worker.wait()  # returns the dismissal value

        if not selected:
            return

        old_watcher_dump = self.fs_search.watcher.model_dump()
        old_watcher_dump["fs_path"] = list(selected)
        watcher = FSWatcher(**old_watcher_dump)
        self.fs_search = FSSearchProvider(
            search_providers=self.fs_search.search_providers,
            watcher=watcher,
        )

        if hasattr(self, "_setup_task"):
            self._setup_task.cancel()

        self._setup_task = asyncio.create_task(self._setup_search())

    def action_close_app(self) -> None:
        """Close the application."""
        self.exit()


def main():
    """Main entry point."""
    app = EverywhereApp()
    app.run()


if __name__ == "__main__":
    main()
