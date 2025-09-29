"""Unified document index for tracking and caching indexed files."""

import os
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from more_itertools import flatten

from ..common.app import app_dirs


@dataclass
class IndexedDocument:
    """Document with metadata and cached processing results."""

    path: Path
    last_modified: float
    size: int
    parsed_text: list[str] | None = None
    embeddings: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    indexed_by: set[str] = field(default_factory=set)

    def is_stale(self) -> bool:
        """Check if document metadata is stale."""
        try:
            stat = self.path.stat()
            return stat.st_mtime != self.last_modified or stat.st_size != self.size
        except (OSError, FileNotFoundError):
            return True

    def is_indexed_by(self, provider_id: str) -> bool:
        """Check if indexed by specific provider."""
        return provider_id in self.indexed_by

    def mark_indexed_by(self, provider_id: str) -> None:
        """Mark as indexed by provider."""
        self.indexed_by.add(provider_id)

    def is_fully_indexed(self, required_providers: set[str]) -> bool:
        """Check if indexed by all required providers."""
        return required_providers.issubset(self.indexed_by)


def remove_children(paths: list[Path]) -> list[Path]:
    """Remove children of the paths."""
    paths = list(set(paths))
    return [p for p in paths if not any(p != p2 and p.is_relative_to(p2) for p2 in paths)]


class DocumentIndex:
    """Unified index for tracking indexed documents and their cached data."""

    def __init__(
        self,
        state_path: Path | None = None,
        path_filter: Callable[[Path], bool] | None = None,
        provider_ids: set[str] | None = None,
    ):
        """Initialize the document index."""
        if state_path is None:
            state_path = app_dirs.app_data_dir / "document_index.pkl"
        self._state_path = Path(state_path)
        self._documents: dict[Path, IndexedDocument] = {}
        self._path_filter = path_filter
        self._provider_ids = provider_ids or set()

        if state_path.exists():
            try:
                self._documents = pickle.loads(state_path.read_bytes())
            except Exception:
                self._documents = {}

    def register_provider(self, provider_id: str) -> None:
        """Register a provider that should index documents."""
        self._provider_ids.add(provider_id)

    @property
    def indexed_directories(self) -> list[Path]:
        """Get indexed directories."""
        return remove_children([doc.path for doc in self._documents.values()])

    def get(self, path: Path) -> IndexedDocument | None:
        """Get document if indexed and not stale."""
        doc = self._documents.get(path)
        if doc is None:
            return None

        if doc.is_stale():
            self._documents.pop(path, None)
            return None

        return doc

    def get_or_create(self, path: Path) -> IndexedDocument:
        """Get existing document or create new one."""
        doc = self.get(path)
        if doc is not None:
            return doc

        try:
            stat = path.stat()
            doc = IndexedDocument(
                path=path,
                last_modified=stat.st_mtime,
                size=stat.st_size,
            )
            self._documents[path] = doc
            return doc
        except (OSError, FileNotFoundError) as e:
            raise ValueError(f"Cannot create document for {path}") from e

    def put(self, doc: IndexedDocument) -> None:
        """Put document in index."""
        self._documents[doc.path] = doc

    def remove(self, path: Path) -> bool:
        """Remove document from index."""
        return self._documents.pop(path, None) is not None

    def compute_diff(self, directories: list[Path]) -> tuple[list[IndexedDocument], list[IndexedDocument]]:
        """Compute what needs indexing vs removal."""
        directories = list(set(directories))

        paths_new: set[Path] = set()
        upserted: list[IndexedDocument] = []
        removed: list[IndexedDocument] = []

        for p in self._walk_all(directories):
            try:
                stat = p.stat()
                paths_new.add(p)

                existing = self._documents.get(p)
                if existing is None or existing.is_stale():
                    doc = IndexedDocument(
                        path=p,
                        last_modified=stat.st_mtime,
                        size=stat.st_size,
                    )
                    self._documents[p] = doc
                    upserted.append(doc)
                elif not existing.is_fully_indexed(self._provider_ids):
                    upserted.append(existing)

            except Exception:
                continue

        for path, doc in list(self._documents.items()):
            if path not in paths_new:
                removed.append(doc)
                # Note: Don't remove from _documents here, let the controller handle it
                # This allows proper cleanup through search providers

        return upserted, removed

    def save(self) -> None:
        """Save the index."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        docs_copy = dict(self._documents)
        self._state_path.write_bytes(pickle.dumps(docs_copy))

    def _walk_all(self, directories: list[Path]) -> Iterable[Path]:
        """Walk all files in directories."""
        scanned_files = flatten([(p for p in fs_dir.rglob("*") if p.is_file()) for fs_dir in directories])
        scanned_files = (p for p in scanned_files if os.access(p, os.R_OK))

        if self._path_filter is not None:
            scanned_files = (p for p in scanned_files if self._path_filter(p))
        return scanned_files
