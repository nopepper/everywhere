"""Basic search provider interface."""

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable

from ..common.pydantic import FrozenBaseModel, SearchQuery, SearchResult


class SearchProvider(FrozenBaseModel, ABC):
    """Search provider."""

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """Supported document types."""

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
