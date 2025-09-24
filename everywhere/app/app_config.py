"""App components."""

from pydantic import BaseModel, Field

from ..search.text_embedding_search import EmbeddingSearchProvider
from ..watchers.fs_watcher import FSWatcher
from .search_controller import SearchController


class AppConfig(BaseModel):
    """App components."""

    embedding_search: EmbeddingSearchProvider = Field(default_factory=EmbeddingSearchProvider)
    fs_watcher: FSWatcher = Field(default_factory=FSWatcher)

    def build_controller(self) -> SearchController:
        """Build the search controller."""
        return SearchController(search_providers=[self.embedding_search], fs_watcher=self.fs_watcher)
