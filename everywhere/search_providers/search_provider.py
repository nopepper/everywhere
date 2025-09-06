"""Basic search provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from ..common.pydantic import FrozenBaseModel
from ..watchers.watcher import WatchEvent

QueryType = TypeVar("QueryType")
ResultType = TypeVar("ResultType")


@dataclass(frozen=True, slots=True)
class SearchQuery(Generic[QueryType]):
    """Search query."""

    text: str


@dataclass(frozen=True, slots=True)
class SearchResult(Generic[ResultType]):
    """Search result."""

    data: ResultType
    confidence: float


class SearchProvider(FrozenBaseModel, ABC, Generic[QueryType, ResultType]):
    """Search provider."""

    @abstractmethod
    def on_change(self, event: WatchEvent[Path]) -> None:
        """Handle a change event."""

    @abstractmethod
    def search(self, query: SearchQuery[QueryType]) -> Iterable[SearchResult[ResultType]]:
        """Search for a query."""

    def setup(self) -> None:
        """Setup the provider."""

    def teardown(self) -> None:
        """Teardown the provider."""
