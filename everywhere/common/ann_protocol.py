"""ANN index protocol for dependency injection."""

from pathlib import Path
from typing import Protocol

import numpy as np


class ANNIndex(Protocol):
    """Protocol for Approximate Nearest Neighbor index."""

    def __contains__(self, path: Path) -> bool:
        """Return whether the index already tracks the given path."""
        ...

    def add(self, path: Path, embedding: np.ndarray) -> None:
        """Store a new embedding for the given path."""
        ...

    def remove(self, path: Path) -> bool:
        """Remove any stored embedding for the path; return ``True`` if removed."""
        ...

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Path, float]]:
        """Return the top ``k`` most similar paths and their scores."""
        ...

    def clean(self) -> int:
        """Purge invalid paths and return how many entries were removed."""
        ...

    def save(self, destination_dir: str | Path | None = None) -> None:
        """Persist the index to ``destination_dir`` or its default location."""
        ...

    def load(self, source_dir: str | Path) -> None:
        """Load index data from ``source_dir``."""
        ...
