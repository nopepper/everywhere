"""ONNX-based text search provider."""

from collections.abc import Iterable
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
from onnxruntime import InferenceSession
from pydantic import Field
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ..common.pydantic import EventType, SearchQuery, SearchResult, WatchEvent
from .search_provider import SearchProvider


# Mean Pooling - Take attention mask into account for correct averaging
def _mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    input_mask_expanded = np.expand_dims(attention_mask, -1) * np.ones_like(token_embeddings)
    return np.sum(token_embeddings * input_mask_expanded, axis=1) / np.clip(
        np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None
    )


def _similarity(obj1: np.ndarray, obj2: np.ndarray) -> float:
    return (float(np.dot(obj1, obj2) / (np.linalg.norm(obj1) * np.linalg.norm(obj2))) + 1) / 2


class ONNXTextSearchProvider(SearchProvider):
    """ONNX Text Embedder."""

    onnx_model_path: Path = Field(description="Path to the ONNX model.")
    tokenizer_path: Path = Field(description="Path to the tokenizer.")
    chunk_size: int = Field(default=2048, description="Chunk size for the text.")

    def setup(self) -> None:
        """Setup the provider."""
        assert self.session is not None
        assert self.tokenizer is not None
        self.embed(["test"])
        self._index: dict[Path, np.ndarray] = {}

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

    def on_change(self, event: WatchEvent) -> None:
        """Handle a change event."""
        if event.event_type == EventType.REMOVED:
            self._index.pop(event.value, None)
        else:
            path = event.value
            if path.suffix not in [".txt", ".md"]:
                return
            text = path.read_text()
            text_chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            self._index[path] = self.embed(text_chunks)

    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search for a query."""
        results = []
        query_embedding = self.embed([query.text])[0]
        for path, embeddings in self._index.items():
            similarity = max(_similarity(query_embedding, emb) for emb in embeddings)
            results.append(SearchResult(value=path, confidence=similarity))
        return results
