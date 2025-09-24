"""Basic filesystem watcher."""

import os
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from more_itertools import flatten

from ..common.app import app_dirs
from ..events.watcher import ChangeType, FileChanged


@dataclass(frozen=True, slots=True)
class PathMeta:
    """Path metadata."""

    path: Path
    last_modified: float
    size: int


def remove_children(paths: list[Path]) -> list[Path]:
    """Remove children of the paths."""
    paths = list(set(paths))
    return [p for p in paths if not any(p.is_relative_to(p2) for p2 in paths)]


class FSIndex:
    """Basic filesystem watcher."""

    def __init__(self, state_path: Path | None = None, path_filter: Callable[[Path], bool] | None = None):
        """Initialize the filesystem watcher."""
        if state_path is None:
            state_path = app_dirs.app_data_dir / "fs_index.pkl"
        self._state_path = Path(state_path)
        self._state: set[PathMeta] = pickle.loads(state_path.read_bytes()) if state_path.exists() else set()
        self._path_filter = path_filter

    @property
    def indexed_directories(self) -> list[Path]:
        """Indexed directories."""
        return remove_children([p.path for p in self._state])

    def update_fs_paths(self, directories: list[Path]) -> Iterable[FileChanged]:
        """Restart the watcher with new paths."""
        directories = list(set(directories))
        new_state: set[PathMeta] = set()

        for p in self.walk_all(directories):
            try:
                p_meta = PathMeta(path=p, last_modified=p.stat().st_mtime, size=p.stat().st_size)
            except Exception:
                # TODO log error
                continue
            new_state.add(p_meta)
            if p_meta not in self._state:
                yield FileChanged(path=p, event_type=ChangeType.UPSERT)

        old_files = {p.path for p in self._state}
        new_files = {p.path for p in new_state}
        removed_paths = old_files - new_files
        for p in removed_paths:
            yield FileChanged(path=p, event_type=ChangeType.REMOVE)
        self._state = new_state

    def save(self) -> None:
        """Save the index."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_bytes(pickle.dumps(self._state))

    def walk_all(self, directories: list[Path]) -> Iterable[Path]:
        """Return all invalidated paths."""
        scanned_files = flatten([(p for p in fs_dir.rglob("*") if p.is_file()) for fs_dir in directories])
        scanned_files = (p for p in scanned_files if os.access(p, os.R_OK))

        if self._path_filter is not None:
            scanned_files = (p for p in scanned_files if self._path_filter(p))
        return scanned_files
