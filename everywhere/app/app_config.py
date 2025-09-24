"""App components."""

from pathlib import Path

from pydantic import BaseModel, Field

from everywhere.index.fs_index import FSIndex

from ..search.text_embedding_search import EmbeddingSearchProvider
from .search_controller import SearchController


class AppConfig(BaseModel):
    """App components."""

    embedding_search: EmbeddingSearchProvider = Field(default_factory=EmbeddingSearchProvider)
    selected_directories: list[Path] = Field(default_factory=list)

    def build_controller(self) -> SearchController:
        """Build the search controller."""
        supported_filetypes = self.embedding_search.parser.supported_types
        index = FSIndex(path_filter=lambda p: p.suffix.strip(".") in supported_filetypes)

        return SearchController(search_providers=[self.embedding_search], fs_index=index)
