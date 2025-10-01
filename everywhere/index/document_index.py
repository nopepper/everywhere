"""Unified document index for tracking and caching indexed files."""

import os
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from more_itertools import flatten

from ..common.app import app_dirs


@dataclass
class IndexedDocument:
    """Document with metadata."""

    path: Path
    last_modified: float
    size: int


def walk_all_files(directories: list[Path], path_filter: Callable[[Path], bool] | None = None) -> Iterable[Path]:
    """Walk all files in directories."""
    scanned_files = flatten([(p for p in fs_dir.rglob("*") if p.is_file()) for fs_dir in directories])
    scanned_files = (p for p in scanned_files if os.access(p, os.R_OK))

    if path_filter is not None:
        scanned_files = (p for p in scanned_files if path_filter(p))
    return scanned_files


class DocumentIndex:
    """SQLite-based index for tracking which providers have indexed which documents."""

    def __init__(
        self,
        db_path: Path | None = None,
    ):
        """Initialize the document index."""
        if db_path is None:
            db_path = app_dirs.app_data_dir / "document_index.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")

        # Create table if it doesn't exist
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_documents (
                path TEXT NOT NULL,
                last_modified REAL NOT NULL,
                size INTEGER NOT NULL,
                provider_id TEXT NOT NULL,
                PRIMARY KEY (path, provider_id)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_path ON indexed_documents(path)
        """)
        self.conn.commit()

    def add(self, path: Path, last_modified: float, size: int, provider_id: str) -> None:
        """Add or update a row in the index."""
        self.conn.execute(
            "INSERT OR REPLACE INTO indexed_documents (path, last_modified, size, provider_id) VALUES (?, ?, ?, ?)",
            (str(path), last_modified, size, provider_id),
        )
        self.conn.commit()

    def remove(self, path: Path, provider_id: str) -> None:
        """Remove a specific provider's index entry for a path."""
        self.conn.execute("DELETE FROM indexed_documents WHERE path = ? AND provider_id = ?", (str(path), provider_id))
        self.conn.commit()

    def get_rows_for_path(self, path: Path) -> list[tuple[float, int, str]]:
        """Get all index entries for a path. Returns list of (last_modified, size, provider_id)."""
        rows = self.conn.execute(
            "SELECT last_modified, size, provider_id FROM indexed_documents WHERE path = ?", (str(path),)
        ).fetchall()
        return rows

    def has_entry(self, path: Path, last_modified: float, size: int, provider_id: str) -> bool:
        """Check if a specific entry exists."""
        row = self.conn.execute(
            "SELECT 1 FROM indexed_documents WHERE path = ? AND last_modified = ? AND size = ? AND provider_id = ?",
            (str(path), last_modified, size, provider_id),
        ).fetchone()
        return row is not None

    def get_all_paths(self) -> set[Path]:
        """Get all unique paths in the index."""
        rows = self.conn.execute("SELECT DISTINCT path FROM indexed_documents").fetchall()
        return {Path(row[0]) for row in rows}

    def save(self) -> None:
        """Commit any pending changes (called for compatibility, but we auto-commit)."""
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
