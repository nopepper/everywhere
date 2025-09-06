"""Basic search provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from ..common.pydantic import FrozenBaseModel

K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True, slots=True)
class SearchQuery(Generic[K]):
    """Search query."""

    text: str


@dataclass(frozen=True, slots=True)
class SearchResult(Generic[V]):
    """Search result."""

    data: V
    confidence: float


class SearchProvider(FrozenBaseModel, ABC, Generic[K, V]):
    """Search provider."""

    @abstractmethod
    def update(self, items: Iterable[tuple[K, V]]) -> None:
        """Update the provider."""

    @abstractmethod
    def search(self, query: SearchQuery[K]) -> Iterable[SearchResult[V]]:
        """Search for a query."""
