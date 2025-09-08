"""ONNX-based text search provider."""

import logging
import threading
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
from more_itertools import unique_everseen
from onnxruntime import InferenceSession
from pydantic import Field
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from voyager import Index, Space, StorageDataType

from ..common.pydantic import SearchQuery, SearchResult
from ..events import add_callback, correlated, publish, release
from ..events.search_provder import (
    IndexingFinished,
    IndexingStarted,
    SetupFinished,
    SetupStarted,
    TeardownFinished,
    TeardownStarted,
)
from ..events.watcher import ChangeType, FileChanged
from .search_provider import SearchProvider


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, -1) * np.ones_like(token_embeddings)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


def _confidence(cos: float, tau: float = 0.3, cap: float = 0.95) -> float:
    if cos <= tau:
        return 0.0
    return max(0.0, min(1.0, (cos - tau) / (cap - tau))) ** 2


class IndexHelper:
    """Index helper."""

    def __init__(self, dims: int):
        """Initialize the index helper."""
        self._index = Index(Space.InnerProduct, num_dimensions=dims, storage_data_type=StorageDataType.E4M3)
        self._id_to_path = {}
        self._path_to_id = {}

    def add(self, path: Path, embedding: np.ndarray) -> None:
        """Add an embedding to the index."""
        new_id = self._index.add_item(embedding)
        self._id_to_path[new_id] = path
        self._path_to_id[path] = new_id
        return new_id

    def remove(self, path: Path) -> bool:
        """Remove an embedding from the index."""
        if path not in self._path_to_id:
            return False
        path_id = self._path_to_id.pop(path)
        del self._id_to_path[path_id]
        self._index.mark_deleted(path_id)
        return True

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Path, float]]:
        """Query the index."""
        ids, distances = self._index.query(embedding, k=min(k, self._index.num_elements))
        assert len(ids.shape) == len(distances.shape) == 1
        return [(self._id_to_path[path_id], distance) for path_id, distance in zip(ids, distances, strict=True)]


class ONNXTextSearchProvider(SearchProvider):
    """ONNX Text Embedder."""

    onnx_model_path: Path = Field(description="Path to the ONNX model.")
    tokenizer_path: Path = Field(description="Path to the tokenizer.")

    k: int = Field(default=100, description="Number of results to return.")
    min_chunk_size: int = Field(default=16, description="Minimum chunk size for the text.")
    max_chunk_size: int = Field(default=2048, description="Maximum chunk size for the text.")
    max_filesize_mb: float = Field(default=10, description="Maximum file size to index in MB.")

    @property
    def supported_types(self) -> list[str]:
        """Supported document types."""
        return ["txt", "md"]

    @correlated
    def setup(self) -> None:
        """Setup the provider."""
        publish(SetupStarted())
        assert self.session is not None
        assert self.tokenizer is not None
        test_emb = self.embed(["test"])
        self._index = IndexHelper(test_emb.shape[-1])
        self._index_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        add_callback(str(id(self)), FileChanged, lambda event: publish(IndexingStarted(path=event.path)))
        add_callback(str(id(self)), FileChanged, self.on_change)
        publish(SetupFinished())

    def teardown(self) -> None:
        """Teardown the provider."""
        publish(TeardownStarted())
        release(str(id(self)))
        publish(TeardownFinished())

    @cached_property
    def session(self) -> InferenceSession:
        """Session."""
        return InferenceSession(self.onnx_model_path)

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """Tokenizer."""
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    def embed(self, text: list[str]) -> np.ndarray:
        """Embed text."""
        batch = cast("dict[str, np.ndarray]", self.tokenizer(text, padding=True, truncation=True, return_tensors="np"))
        embs = self.session.run(
            None,
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "token_type_ids": batch["token_type_ids"],
            },
        )[0]
        embs = cast("np.ndarray", embs)
        if len(embs.shape) == 1:
            embs = embs.reshape(1, -1)
        return _mean_pooling(embs, batch["attention_mask"])

    @correlated
    def on_change(self, event: FileChanged) -> None:
        """Handle a change event."""
        success = False
        try:
            if event.event_type == ChangeType.REMOVED:
                with self._index_lock:
                    self._index.remove(event.path)
            else:
                path = event.path
                if path.suffix.strip(".") not in self.supported_types:
                    return
                if path.stat().st_size > self.max_filesize_mb * 1024 * 1024:
                    return

                # Try to read with UTF-8 first, then fallback to other encodings
                text = None
                encodings = ["utf-8", "utf-16", "latin-1", "cp1252"]

                for encoding in encodings:
                    try:
                        text = path.read_text(encoding=encoding)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue

                if text is None:
                    # If all encodings fail, skip this file
                    logging.warning(f"Could not decode file {path} with any supported encoding. Skipping file.")
                    return

                text_chunks = [text[i : i + self.max_chunk_size] for i in range(0, len(text), self.max_chunk_size)]
                text_chunks = [chunk for chunk in text_chunks if len(chunk) >= self.min_chunk_size]
                if len(text_chunks) == 0:
                    return
                for chunk in text_chunks:
                    self._idle.wait()
                    emb = self.embed([chunk])[0]
                    emb = emb / np.linalg.norm(emb)
                    self._index.add(path, emb)
                success = True
        finally:
            publish(IndexingFinished(success=success, path=path))

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results = []
        self._idle.clear()
        query_embedding = self.embed([query.text])[0]
        with self._index_lock:
            for path, distance in self._index.query(query_embedding, self.k):
                similarity = 1 - distance
                results.append(SearchResult(value=path, confidence=_confidence(similarity)))
        results.sort(key=lambda x: x.confidence, reverse=True)
        results = list(unique_everseen(results, key=lambda x: x.value))
        self._idle.set()
        return results
