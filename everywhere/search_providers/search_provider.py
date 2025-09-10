"""Basic search provider interface."""

import hashlib
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from queue import SimpleQueue

from ..common.pydantic import FrozenBaseModel, SearchQuery, SearchResult
from ..events import add_callback, correlated, publish
from ..events.app import UserSearched
from ..events.search_provder import (
    GotIndexingRequest,
    GotSearchResult,
    IndexingFinished,
    IndexingStarted,
    SearchFinished,
    SearchStarted,
)
from ..events.watcher import FileChanged

_STOP = object()


class _EventfulProvider:
    """Event-wrapped provider with 'latest search wins' semantics."""

    def __init__(self, parent: "SearchProvider"):
        self.parent = parent
        self.parent.setup()

        # file update worker
        self._task_q: SimpleQueue = SimpleQueue()
        self._task_thread = threading.Thread(target=self._task_loop, daemon=True)
        self._task_thread.start()

        # search worker
        self._search_q: SimpleQueue = SimpleQueue()
        self._search_abort = threading.Event()
        self._search_thread = threading.Thread(target=self._search_loop, daemon=True)
        self._search_thread.start()

        # callbacks (keep handles if you have remove_callback)
        add_callback(FileChanged, self.on_file_changed)
        add_callback(UserSearched, self.on_user_searched)

    def _task_loop(self) -> None:
        while True:
            item = self._task_q.get()
            if item is _STOP:
                break
            path = item
            publish(IndexingStarted(path=path))
            try:
                self.parent.update(path)
                publish(IndexingFinished(success=True, path=path))
            except Exception:
                publish(IndexingFinished(success=False, path=path))
                raise
                # optionally log/trace here

    def _search_loop(self) -> None:
        while True:
            query = self._search_q.get()
            if query is _STOP:
                return
            self._search_abort.clear()
            self._search(query)

    @correlated
    def _search(self, query: str):
        publish(SearchStarted(query=query))
        try:
            for result in self.parent.search(SearchQuery(text=query)):
                if self._search_abort.is_set():
                    break
                publish(GotSearchResult(query=query, result=result))
        finally:
            publish(SearchFinished(query=query))

    @correlated
    def teardown(self) -> None:
        self._task_q.put(_STOP)
        self._task_thread.join()

        self._search_abort.set()
        self._search_q.put(_STOP)
        self._search_thread.join()

        self.parent.teardown()

    @correlated
    def on_file_changed(self, event: FileChanged) -> None:
        publish(GotIndexingRequest(path=event.path))
        self._task_q.put(event.path)

    @correlated
    def on_user_searched(self, event: UserSearched) -> None:
        self._search_abort.set()
        self._search_q.put(event.query)


class SearchProvider(FrozenBaseModel, ABC):
    """Search provider."""

    @property
    @abstractmethod
    def supported_types(self) -> set[str]:
        """Supported document types."""

    @abstractmethod
    def update(self, path: Path) -> bool:
        """Update the provider."""

    @abstractmethod
    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""

    def setup(self) -> None:
        """Setup the provider."""

    def teardown(self) -> None:
        """Teardown the provider."""

    @property
    def provider_id(self) -> str:
        """Provider ID."""
        return hashlib.md5((str(type(self)) + "\n" + self.model_dump_json()).encode()).hexdigest()

    def start_eventful(self) -> _EventfulProvider:
        """Start the eventful provider."""
        return _EventfulProvider(self)
