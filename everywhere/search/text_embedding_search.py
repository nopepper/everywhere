"""ONNX-based text search provider."""

import threading
from itertools import groupby
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from ..common.ann import ANNIndex
from ..common.app import app_dirs
from ..common.pydantic import SearchQuery, SearchResult
from ..events import add_callback, publish
from ..events.app import UserSearched
from ..events.search_provder import GotSearchResult, IndexingFinished, IndexingStarted
from ..events.watcher import ChangeType, FileChanged
from .search_provider import SearchProviderService
from .text.chunking import TextChunker
from .text.onnx_embedder import ONNXEmbedder
from .text.parsing import TextParser


class EmbeddingSearchProvider(BaseModel, SearchProviderService):
    """ONNX Text Embedder."""

    embedder: ONNXEmbedder = Field(default_factory=ONNXEmbedder)
    parser: TextParser = Field(default_factory=TextParser)
    chunker: TextChunker = Field(default_factory=TextChunker)

    k: int = Field(default=1000, description="Number of results to return.")
    ann_cache_dir: Path = Field(
        default_factory=lambda: app_dirs.app_cache_dir / "text_ann", description="Path to the ANN cache directory."
    )

    def start(self) -> None:
        """Post init."""
        test_emb = self.embedder.embed(["test"])
        self._index = ANNIndex(dims=test_emb.shape[-1], cache_dir=self.ann_cache_dir).start_eventful()
        self._idle = threading.Event()
        self._idle.set()
        add_callback(FileChanged, self.handle_file_change)
        add_callback(UserSearched, self.on_user_searched, skip_old=True)

    def stop(self) -> None:
        """Stop the search provider."""
        self._idle.set()
        self._index.save()

    def on_user_searched(self, event: UserSearched) -> None:
        """Handle a user searched event."""
        for result in self.search(SearchQuery(text=event.query)):
            publish(GotSearchResult(query=event.query, result=result))

    def handle_file_change(self, event: FileChanged) -> None:
        """Handle a change event."""
        publish(IndexingStarted(path=event.path))
        try:
            result = self._on_file_changed(event)
        except Exception:
            result = False
        finally:
            publish(IndexingFinished(path=event.path, success=result))

    def _on_file_changed(self, event: FileChanged) -> bool:
        """Handle a change event."""
        path = event.path

        if path in self._index:
            return False

        if not path.exists() or event.event_type == ChangeType.DELETE:
            return self._index.remove(path)

        text_chunks = self.chunker.chunk(self.parser.parse(path))

        if not text_chunks:
            return False

        for chunk in text_chunks:
            self._idle.wait()
            emb = self.embedder.embed([chunk])[0]
            emb = emb / np.linalg.norm(emb)
            self._index.add(path, emb)
        return True

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        self._idle.clear()
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
            chunk_score = np.mean([x.confidence for x in group]).item()
            # num_chunks = len(self._index._index_helper._ids_by_path[path.as_posix()])
            doc_score = chunk_score  # / math.log(1 + 0.1 * num_chunks)
            results_filtered.append(SearchResult(value=path, confidence=doc_score))

        self._idle.set()
        return results_filtered
