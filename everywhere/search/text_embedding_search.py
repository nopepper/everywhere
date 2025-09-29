"""ONNX-based text search provider."""

from itertools import groupby
from pathlib import Path
from typing import Any, Self

import numpy as np
from pydantic import BaseModel, Field

from ..common.app import app_dirs
from ..common.pydantic import SearchQuery, SearchResult
from ..events import publish
from ..events.search_provider import IndexingFinished, IndexingStarted
from ..index.ann import ANNIndex
from ..index.document_index import IndexedDocument
from .search_provider import SearchProvider
from .text.chunking import TextChunker
from .text.onnx_embedder import ONNXEmbedder
from .text.parsing import TextParser


class EmbeddingSearchProvider(BaseModel, SearchProvider):
    """ONNX Text Embedder."""

    embedder: ONNXEmbedder = Field(default_factory=ONNXEmbedder)
    parser: TextParser = Field(default_factory=TextParser)
    chunker: TextChunker = Field(default_factory=TextChunker)

    k: int = Field(default=1000, description="Number of results to return.")
    ann_cache_dir: Path = Field(
        default_factory=lambda: app_dirs.app_cache_dir / "text_ann", description="Path to the ANN cache directory."
    )

    def __enter__(self) -> Self:
        """Post init."""
        test_emb = self.embedder.embed(["test"])
        self._index = ANNIndex(dims=test_emb.shape[-1], cache_dir=self.ann_cache_dir)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Stop the search provider."""
        self._index.save()

    def index_document(self, doc: IndexedDocument) -> bool:
        """Index a document."""
        publish(IndexingStarted(path=doc.path))
        try:
            if doc.parsed_text:
                text_chunks = self.chunker.chunk(doc.parsed_text)
            else:
                parsed_text = self.parser.parse(doc.path)
                text_chunks = self.chunker.chunk(parsed_text)
                doc.parsed_text = parsed_text

            if not text_chunks:
                return False

            for chunk in text_chunks:
                emb = self.embedder.embed([chunk])[0]
                emb = emb / np.linalg.norm(emb)
                self._index.add(doc.path, emb)

            return True
        except Exception:
            return False
        finally:
            publish(IndexingFinished(path=doc.path, success=True))

    def remove_document(self, path: Path) -> bool:
        """Remove a document from the embedding index."""
        return self._index.remove(path)

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results: list[SearchResult] = []
        if len(query.text) == 0:
            return []

        query_embedding = self.embedder.embed([query.text])[0]
        # Normalize query to align with normalized corpus vectors
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        for path, distance in self._index.query(query_embedding, self.k):
            similarity = 1 - distance
            results.append(SearchResult(value=path, confidence=similarity))

        results_filtered: list[SearchResult] = []
        for path, group in groupby(sorted(results, key=lambda x: x.value), key=lambda x: x.value):
            doc_score = np.mean([x.confidence for x in group]).item()
            results_filtered.append(SearchResult(value=path, confidence=doc_score))

        return results_filtered
