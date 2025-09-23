"""Search bar widget with debounced event emission."""

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Input, Static

from ...common.debounce import DebouncedRunner
from ...events import add_callback, publish
from ...events.app import UserSearched
from ...events.search_provder import IndexingFinished

DEBOUNCE_LATENCY = 0.1


class SearchBar(Static):
    """A search bar widget that emits debounced search events."""

    def __init__(
        self,
        placeholder: str = "Search files and folders...",
        debounce_latency: float = DEBOUNCE_LATENCY,
        **kwargs: Any,
    ):
        """Initialize the search bar."""
        super().__init__(**kwargs)
        self.input = Input(placeholder=placeholder, classes="search-input", id="search_input")
        self._debounced = DebouncedRunner(debounce_latency)

    def compose(self) -> ComposeResult:
        """Compose the search bar."""
        yield self.input

    def on_mount(self) -> None:
        """Mount the search bar."""
        self.input.focus()
        add_callback(IndexingFinished, self._on_indexing_finished)

    def _on_indexing_finished(self, event: IndexingFinished) -> None:
        """Handle indexing finished event."""
        self._debounced.submit(lambda: publish(UserSearched(query=self.input.value)))

    def on_input_changed(self, message: Input.Changed) -> None:
        """Handle input changed event."""
        if message.input.id != "search_input":
            return
        self._debounced.submit(lambda: publish(UserSearched(query=message.value)))
