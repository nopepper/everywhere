"""App components."""

from pathlib import Path

from pydantic import BaseModel, Field

from everywhere.index.document_index import DocumentIndex

from ..search.tantivy_search import TantivySearchProvider
from ..search.text_embedding_search import EmbeddingSearchProvider
from .search_controller import SearchController


class AppConfig(BaseModel):
    """App components."""

    embedding_search: EmbeddingSearchProvider = Field(default_factory=EmbeddingSearchProvider)
    tantivy_search: TantivySearchProvider = Field(default_factory=TantivySearchProvider)
    selected_directories: list[Path] = Field(default_factory=list)


def build_controller(config: AppConfig) -> SearchController:
    """Build the search controller."""
    supported_filetypes = config.embedding_search.parser.supported_types | config.tantivy_search.parser.supported_types

    def path_filter(p: Path) -> bool:
        return p.suffix.strip(".") in supported_filetypes

    doc_index = DocumentIndex()

    return SearchController(
        search_providers=[config.embedding_search, config.tantivy_search],
        doc_index=doc_index,
        path_filter=path_filter,
    )
