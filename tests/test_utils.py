"""Test utilities and fake implementations."""

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
from markitdown import StreamInfo

from everywhere.common.ann_protocol import ANNIndex
from everywhere.common.clock import Clock, Timer
from everywhere.common.pydantic import SearchQuery, SearchResult
from everywhere.search_providers.document_extractor import DocumentExtractor
from everywhere.search_providers.embedding_backend import EmbeddingBackend
from everywhere.search_providers.onnx_text_search import ONNXTextSearchProvider
from everywhere.search_providers.search_provider import SearchProvider
from everywhere.watchers.fs_watcher import FSWatcher


class FakeClock(Clock):
    """Fake clock that executes timers immediately."""

    def timer(self, delay: float, callback: Callable[[], None]) -> Timer:
        """Create a timer that executes immediately."""
        return FakeTimer(delay, callback)


class FakeTimer(Timer):
    """Fake timer that executes immediately."""

    def start(self) -> None:
        """Execute the callback immediately."""
        if not self._cancelled:
            self.callback()


class FakeEmbeddingBackend(EmbeddingBackend):
    """Fake embedding backend for testing."""

    def __init__(self, dims: int = 384):
        """Store embedding dimensionality for generated vectors."""
        self._dims = dims

    @property
    def dims(self) -> int:
        """Return the dimensionality used for embeddings."""
        return self._dims

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Create a deterministic but different embedding for each text
            hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(hash_val)
            emb = rng.normal(0.0, 1.0, self._dims).astype(np.float32)
            norm = np.linalg.norm(emb)
            emb = (
                np.ones(self._dims, dtype=np.float32)
                if norm == 0
                else emb / norm  # Normalize
            )
            embeddings.append(emb)
        return np.array(embeddings)


class FakeDocumentExtractor(DocumentExtractor):
    """Fake document extractor for testing."""

    def __init__(self, text_map: dict[Path, str] | None = None):
        """Initialize with a mapping of paths to mock document content."""
        self.text_map = text_map or {}

    def extract(self, path: Path, stream_info: StreamInfo | None = None) -> str:
        """Return fake text for the given path."""
        return self.text_map.get(path, f"Fake content for {path}")


class InMemoryANNIndex:
    """In-memory ANN index implementation for testing."""

    def __init__(self, dims: int):
        """Initialize storage for embeddings with the given dimensionality."""
        self._dims = dims
        self._embeddings: dict[Path, np.ndarray] = {}
        self._paths: list[Path] = []

    def __contains__(self, path: Path) -> bool:
        """Return whether an embedding exists for ``path``."""
        return path in self._embeddings

    def add(self, path: Path, embedding: np.ndarray) -> None:
        """Insert or update the stored embedding for ``path``."""
        if path not in self._embeddings:
            self._paths.append(path)
        self._embeddings[path] = embedding

    def remove(self, path: Path) -> bool:
        """Delete the embedding for ``path`` if it exists."""
        if path in self._embeddings:
            del self._embeddings[path]
            self._paths.remove(path)
            return True
        return False

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Path, float]]:
        """Simple cosine similarity search."""
        results = []
        for path in self._paths:
            emb = self._embeddings[path]
            similarity = float(np.dot(embedding, emb))
            results.append((path, -similarity))  # Negative for distance-like behavior

        results.sort(key=lambda x: x[1])  # Sort by distance (ascending)
        return [(path, -dist) for path, dist in results[:k]]  # Convert back to similarity

    def clean(self) -> int:
        """Clean invalid paths (always returns 0 for fake)."""
        return 0

    def save(self, destination_dir: str | Path | None = None) -> None:
        """No-op for in-memory implementation."""

    def load(self, source_dir: str | Path) -> None:
        """No-op for in-memory implementation."""


def create_fake_ann(dims: int = 384) -> ANNIndex:
    """Create a fake ANN index for testing."""
    return InMemoryANNIndex(dims)


class FakeTextSearchProvider(SearchProvider):
    """Simple text search provider that indexes files by substring matching."""

    max_filesize_mb: float = 10.0
    k: int = 10

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        """Initialize in-memory storage after Pydantic construction."""
        self._documents: dict[Path, str] = {}

    @property
    def supported_types(self) -> set[str]:
        """Return file suffixes accepted by the fake provider."""
        return {"txt", "md", "py"}

    def setup(self) -> None:
        """Prepare the provider state (no-op for fake)."""
        self._documents.clear()

    def teardown(self) -> None:
        """Reset provider state (no-op for fake)."""
        self._documents.clear()

    def update(self, path: Path) -> bool:
        """Index or remove a path if it matches supported types."""
        if path.suffix.strip(".").lower() not in self.supported_types:
            return False

        if not path.exists():
            self._documents.pop(path, None)
            return True

        max_bytes = self.max_filesize_mb * 1024 * 1024
        if path.stat().st_size > max_bytes:
            return False

        self._documents[path] = path.read_text(encoding="utf-8")
        return True

    def search(self, query: SearchQuery) -> list[SearchResult]:
        """Return matching results ordered by naive confidence."""
        text = query.text.strip().lower()
        if not text:
            return []

        matches: list[SearchResult] = []
        for path, content in self._documents.items():
            haystack = content.lower()
            occurrences = haystack.count(text)
            if occurrences == 0:
                continue

            confidence = min(1.0, occurrences / max(1, len(haystack)))
            matches.append(SearchResult(value=path, confidence=confidence))

        matches.sort(key=lambda result: result.confidence, reverse=True)
        return matches[: self.k]


class StubONNXTextSearchProvider(ONNXTextSearchProvider):
    """Stub implementation of the ONNX search provider for testing."""

    max_filesize_mb: float = 10.0
    k: int = 10

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        """Wrap an in-memory fake provider for reuse in tests."""
        self._fake_provider = FakeTextSearchProvider(max_filesize_mb=self.max_filesize_mb, k=self.k)

    @property
    def supported_types(self) -> set[str]:
        """Expose supported types from the underlying fake provider."""
        return self._fake_provider.supported_types

    def setup(self) -> None:
        """Set up the fake provider with mirrored configuration."""
        self._fake_provider.max_filesize_mb = self.max_filesize_mb
        self._fake_provider.k = self.k
        self._fake_provider.setup()

    def teardown(self) -> None:
        """Delegate teardown to the fake provider."""
        self._fake_provider.teardown()

    def update(self, path: Path) -> bool:
        """Update using the fake provider implementation."""
        return self._fake_provider.update(path)

    def search(self, query: SearchQuery) -> Iterable[SearchResult]:
        """Search using the fake provider's index."""
        return self._fake_provider.search(query)


class FakeFSWatcher(FSWatcher):
    """FS watcher that scans directories synchronously without background threads."""

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        """Initialize internal running flag."""
        self._running = False

    def start(self) -> None:  # type: ignore[override]
        """Mark the watcher as started without spawning threads."""
        self._running = True

    def stop(self) -> None:  # type: ignore[override]
        """Mark the watcher as stopped."""
        self._running = False

    def drain(self) -> list[Path]:
        """Return all files currently tracked by the watcher."""
        return [path for path in self.walk_all() if path.is_file()]
