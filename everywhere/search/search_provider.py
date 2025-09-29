"""Basic search provider interface."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Self

from ..common.pydantic import SearchQuery, SearchResult
from ..index.document_index import IndexedDocument


class SearchProvider(ABC):
    """Search provider service."""

    @property
    def provider_id(self) -> str:
        """Unique identifier for this provider."""
        return type(self).__name__

    @abstractmethod
    def __enter__(self) -> Self:
        """Start the search provider."""

    @abstractmethod
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Stop the search provider."""

    @abstractmethod
    def index_document(self, doc: IndexedDocument) -> bool:
        """Index or update a document. Return True if successful."""

    @abstractmethod
    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""
