"""App components."""

from pathlib import Path

from pydantic import BaseModel, Field

from everywhere.index.document_index import DocumentIndex

from ..search.text_embedding_search import EmbeddingSearchProvider
from .search_controller import SearchController


class AppConfig(BaseModel):
    """App components."""

    embedding_search: EmbeddingSearchProvider = Field(default_factory=EmbeddingSearchProvider)
    selected_directories: list[Path] = Field(default_factory=list)


def build_controller(config: AppConfig) -> SearchController:
    """Build the search controller."""
    supported_filetypes = config.embedding_search.parser.supported_types
    doc_index = DocumentIndex(path_filter=lambda p: p.suffix.strip(".") in supported_filetypes)

    return SearchController(search_providers=[config.embedding_search], doc_index=doc_index)
