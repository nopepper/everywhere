"""Search bar widget that triggers debounced searches through the app."""

from typing import Any

from textual.app import ComposeResult
from textual.message import Message
from textual.widgets import Input, Static

from ...events import add_callback
from ...events.search_provider import IndexingFinished


class SearchBar(Static):
    """A search bar widget that emits debounced search events."""

    class SearchTriggered(Message):
        """Message emitted when the search input changes."""

        def __init__(self, query: str) -> None:
            """Initialize the search triggered message."""
            super().__init__()
            self.query = query

    def __init__(
        self,
        placeholder: str = "Search files and folders...",
        **kwargs: Any,
    ):
        """Initialize the search bar."""
        super().__init__(**kwargs)
        self.input = Input(placeholder=placeholder, classes="search-input", id="search_input")

    def compose(self) -> ComposeResult:
        """Compose the search bar."""
        yield self.input

    def on_mount(self) -> None:
        """Mount the search bar."""
        self.input.focus()
        add_callback(IndexingFinished, self._on_indexing_finished)

    def _on_indexing_finished(self, event: IndexingFinished) -> None:
        """Handle indexing finished event."""
        self._trigger_search(self.input.value)

    def on_input_changed(self, message: Input.Changed) -> None:
        """Handle input changed event."""
        if message.input.id != "search_input":
            return
        self._trigger_search(message.value)

    def _trigger_search(self, query: str) -> None:
        """Request the app to perform a debounced search."""
        self.post_message(self.SearchTriggered(query))
