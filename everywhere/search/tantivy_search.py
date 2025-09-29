"""Tantivy-based full-text search provider."""

from pathlib import Path
from typing import Any, Self

import tantivy
from pydantic import BaseModel, Field

from ..common.app import app_dirs
from ..common.pydantic import SearchQuery, SearchResult
from ..events import publish
from ..events.search_provider import IndexingFinished, IndexingStarted
from ..index.document_index import IndexedDocument
from .search_provider import SearchProvider
from .text.parsing import TextParser


class TantivySearchProvider(BaseModel, SearchProvider):
    """Full-text search provider using Tantivy."""

    parser: TextParser = Field(default_factory=TextParser)
    index_dir: Path = Field(
        default_factory=lambda: app_dirs.app_cache_dir / "tantivy_index",
        description="Path to the Tantivy index directory.",
    )
    k: int = Field(default=1000, description="Maximum number of results to return.")

    def __enter__(self) -> Self:
        """Initialize the Tantivy index."""
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Define schema
        # Note: path must be a STRING field (not TEXT) for reliable deletion
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("path", stored=True, tokenizer_name="raw")
        schema_builder.add_text_field("content", stored=False)
        self._schema = schema_builder.build()

        # Create or open index
        try:
            self._index = tantivy.Index.open(str(self.index_dir))
        except Exception:
            self._index = tantivy.Index(self._schema, path=str(self.index_dir))

        self._writer = self._index.writer()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Close the search provider."""
        if hasattr(self, "_writer"):
            self._writer.commit()
            del self._writer

    def index_document(self, doc: IndexedDocument) -> bool:
        """Index a document in Tantivy."""
        publish(IndexingStarted(path=doc.path))
        success = False

        try:
            # Parse text from document
            if doc.parsed_text:
                text_content = " ".join(doc.parsed_text)
            else:
                parsed_text = self.parser.parse(doc.path)
                text_content = " ".join(parsed_text)
                doc.parsed_text = parsed_text

            if not text_content.strip():
                return False

            # Create document
            tantivy_doc = tantivy.Document()
            tantivy_doc.add_text("path", str(doc.path))
            tantivy_doc.add_text("content", text_content)

            # Add to index
            self._writer.add_document(tantivy_doc)
            success = True
            return True

        except Exception:
            return False

        finally:
            publish(IndexingFinished(path=doc.path, success=success))

    def remove_document(self, path: Path) -> bool:
        """Remove a document from the Tantivy index."""
        try:
            # Delete documents with matching path field
            self._writer.delete_documents("path", str(path))
            return True
        except Exception:
            return False

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search using Tantivy's full-text search."""
        if not query.text.strip():
            return []

        try:
            # Commit any pending changes and reload index
            self._writer.commit()
            self._index.reload()

            # Get searcher
            searcher = self._index.searcher()

            # Parse and execute query
            tantivy_query = self._index.parse_query(query.text, ["content"])
            search_results = searcher.search(tantivy_query, self.k)

            # Convert results
            results: list[SearchResult] = []

            for score, doc_address in search_results.hits:
                doc = searcher.doc(doc_address)
                doc_dict = doc.to_dict()
                path_values = doc_dict.get("path", [])
                if path_values:
                    path_str = path_values[0]
                    # Normalize score to 0-1 range (BM25 scores can be arbitrary)
                    normalized_score = min(1.0, score / 10.0)
                    results.append(SearchResult(value=Path(path_str), confidence=normalized_score))

            return results

        except Exception:
            # Query parsing or search failed
            return []
