"""ONNX-based text search provider."""

import logging
import os
import threading
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
from more_itertools import unique_everseen
from pydantic import Field
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from voyager import Index, Space, StorageDataType

from ..common.pydantic import SearchQuery, SearchResult
from ..events import add_callback, correlated, publish
from ..events.search_provder import (
    IndexingFinished,
    IndexingStarted,
)
from ..events.watcher import ChangeType, FileChanged
from .search_provider import SearchProvider


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, -1) * np.ones_like(token_embeddings)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


class IndexHelper:
    """Index helper."""

    def __init__(self, dims: int):
        """Initialize the index helper."""
        self._index = Index(Space.InnerProduct, num_dimensions=dims, storage_data_type=StorageDataType.E4M3)
        self._tracked_paths = []

    def add(self, path: Path, embedding: np.ndarray) -> None:
        """Add an embedding to the index."""
        if len(embedding.shape) == 1:
            new_id = self._index.add_item(embedding)
            self._tracked_paths.append(
                {
                    "path": path,
                    "id": new_id,
                }
            )
        else:
            assert len(embedding.shape) == 2
            new_ids = self._index.add_items(embedding)
            for new_id in new_ids:
                self._tracked_paths.append(
                    {
                        "path": path,
                        "id": new_id,
                    }
                )
        return new_id

    def remove(self, path: Path) -> bool:
        """Remove an embedding from the index."""
        for row in self._tracked_paths:
            if row["path"] == path:
                self._index.mark_deleted(row["id"])
                self._tracked_paths.remove(row)
                return True
        return False

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Path, float]]:
        """Query the index."""
        ids, distances = self._index.query(embedding, k=min(k, self._index.num_elements))
        assert len(ids.shape) == len(distances.shape) == 1
        results = []
        for path_id, distance in zip(ids, distances, strict=True):
            path = next((row["path"] for row in self._tracked_paths if row["id"] == path_id), None)
            if path is not None:
                results.append((path, distance))
        return results


class ONNXTextSearchProvider(SearchProvider):
    """ONNX Text Embedder."""

    onnx_model_path: Path = Field(description="Path to the ONNX model.")
    tokenizer_path: Path = Field(description="Path to the tokenizer.")

    k: int = Field(default=1000, description="Number of results to return.")
    min_chunk_size: int = Field(default=16, description="Minimum chunk size for the text.")
    max_chunk_size: int = Field(default=2048, description="Maximum chunk size for the text.")
    overlap: int = Field(default=128, description="Overlap for the text chunks.")
    max_filesize_mb: float = Field(default=10, description="Maximum file size to index in MB.")

    @property
    def supported_types(self) -> list[str]:
        """Supported document types."""
        return ["txt", "md"]

    @correlated
    def setup(self) -> None:
        """Setup the provider."""
        assert self.session is not None
        assert self.tokenizer is not None
        test_emb = self.embed(["test"])
        self._index = IndexHelper(test_emb.shape[-1])
        self._index_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        add_callback(FileChanged, lambda event: publish(IndexingStarted(path=event.path)))
        add_callback(FileChanged, self.on_change)

    def teardown(self) -> None:
        """Teardown the provider."""

    @cached_property
    def session(self) -> ort.InferenceSession:
        """Session."""
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        cpu_count = max(1, (os.cpu_count() or 4))
        options.intra_op_num_threads = max(1, cpu_count - 2)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        return ort.InferenceSession(self.onnx_model_path, sess_options=options, providers=["CPUExecutionProvider"])

    @cached_property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        """Tokenizer."""
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    def embed(self, text: list[str]) -> np.ndarray:
        """Embed text."""
        # Fixed-length padding helps performance in ORT by producing uniform shapes
        max_len = min(getattr(self.tokenizer, "model_max_length", 512) or 512, 512)
        batch = cast(
            "dict[str, np.ndarray]",
            self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="np",
            ),
        )
        embs = self.session.run(
            None,
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "token_type_ids": batch.get("token_type_ids", np.zeros_like(batch["input_ids"])),
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

                # Create overlapping chunks with 20% overlap
                step_size = self.max_chunk_size - self.overlap
                text_chunks = [text[i : i + self.max_chunk_size] for i in range(0, len(text), step_size)]
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
            publish(IndexingFinished(success=success, path=event.path))

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Search for a query."""
        results = []
        self._idle.clear()
        query_embedding = self.embed([query.text])[0]
        # Normalize query to align with normalized corpus vectors
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        with self._index_lock:
            for path, distance in self._index.query(query_embedding, self.k):
                similarity = 1 - distance
                results.append(SearchResult(value=path, confidence=similarity))
        results.sort(key=lambda x: x.confidence, reverse=True)
        results = list(unique_everseen(results, key=lambda x: x.value))
        self._idle.set()
        return results
