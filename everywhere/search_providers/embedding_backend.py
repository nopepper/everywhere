"""Embedding backend protocol for dependency injection."""

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    """Protocol for text embedding backends."""

    @property
    def dims(self) -> int:
        """Return the dimensionality of produced embeddings."""
        ...

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return embeddings for the supplied ``texts``."""
        ...
