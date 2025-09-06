"""Application entry point."""

import asyncio
from datetime import datetime
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widgets import DataTable, Header, Input

from ..common.pydantic import SearchResult
from ..search_providers.fs_search import FSSearchProvider
from ..search_providers.onnx_text_search import ONNXTextSearchProvider
from ..search_providers.search_provider import SearchQuery
from ..watchers.fs_watcher import FSWatcher


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


class EverywhereApp(App):
    """File search application."""

    CSS = """
    Input {
        dock: top;
        height: 3;
        margin: 1;
    }

    DataTable {
        height: 1fr;
        margin: 0 1;
    }

    .search-input {
        border: solid $accent;
    }
    """

    search_term = reactive("")

    def __init__(self, fs_path: str = "data_test"):
        """Initialize the Everything app.

        Args:
            fs_path: Path to the directory to search in.
        """
        super().__init__()

        # Initialize search system with hardcoded values from notebook
        self.watcher = FSWatcher(
            fs_path=Path(fs_path),
        )
        self.text_search = ONNXTextSearchProvider(
            onnx_model_path=Path("models/all-MiniLM-L6-v2/onnx/model_quint8_avx2.onnx"),
            tokenizer_path=Path("models/all-MiniLM-L6-v2"),
        )
        self.fs_search = FSSearchProvider(
            search_providers=[self.text_search],
            watcher=self.watcher,
        )
        self.search_setup_done = False

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()
        yield Input(placeholder="Search files and folders...", classes="search-input", id="search_input")
        yield DataTable(id="results_table")

    def on_mount(self) -> None:
        """Set up the app when mounted."""
        # Set up the results table
        table = self.query_one("#results_table", DataTable)
        table.add_columns("Name", "Path", "Size", "Date Modified")
        table.cursor_type = "cell"

        # Initialize search system in background
        self._setup_task = asyncio.create_task(self._setup_search())

    async def _setup_search(self) -> None:
        """Set up the search system asynchronously."""
        try:
            # This might take some time, so do it in background
            await asyncio.get_event_loop().run_in_executor(None, self.fs_search.setup)
            self.search_setup_done = True
            # Trigger initial search if there's already a search term
            if self.search_term:
                await self._perform_search()
        except Exception as e:
            self.notify(f"Failed to initialize search: {e}", severity="error")

    def on_input_changed(self, message: Input.Changed) -> None:
        """Handle search input changes."""
        if message.input.id == "search_input":
            self.search_term = message.value
            # Create task for search to avoid blocking UI
            self._search_task = asyncio.create_task(self._perform_search())

    async def _perform_search(self) -> None:
        """Perform search and update results."""
        if not self.search_setup_done or not self.search_term.strip():
            # Clear results if no search term
            table = self.query_one("#results_table", DataTable)
            table.clear()
            return

        try:
            # Perform search in background thread to avoid blocking UI
            search_query = SearchQuery(text=self.search_term)
            results = await asyncio.get_event_loop().run_in_executor(None, list, self.fs_search.search(search_query))

            # TODO: don't hardcode this
            results = results[:1000]

            # Update the table with results
            await self._update_results_table(results)

        except Exception as e:
            self.notify(f"Search error: {e}", severity="error")

    async def _update_results_table(self, results: list[SearchResult]) -> None:
        """Update the results table with search results."""
        table = self.query_one("#results_table", DataTable)
        table.clear()

        if not results:
            # Add a ghost result with 0 confidence when no results found
            confidence_label = confidence_to_color(0.0)
            table.add_row("", "", "", "", label=confidence_label)
            return

        for result in results:
            path = result.value
            confidence_label = confidence_to_color(result.confidence)
            try:
                stat = path.stat()
                name = path.name
                path_str = str(path)
                size = format_size(stat.st_size)
                date_modified = format_date(stat.st_mtime_ns)

                table.add_row(name, path_str, size, date_modified, label=confidence_label)
            except (OSError, FileNotFoundError):
                # Handle case where file might have been deleted
                table.add_row(path.name, str(path), "N/A", "N/A", label=confidence_label)


def main():
    """Main entry point."""
    app = EverywhereApp()
    app.run()


if __name__ == "__main__":
    main()
