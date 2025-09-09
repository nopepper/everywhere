"""Basic filesystem watcher."""

import os
import threading
from collections.abc import Iterable
from pathlib import Path

from more_itertools import flatten
from pydantic import Field, field_validator
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..common.pydantic import FrozenBaseModel
from ..events import publish
from ..events.watcher import ChangeType, FileChanged


class _FSWatchdogEventHandler(FileSystemEventHandler):
    def __init__(self, fs_path: Path):
        """Initialize the filesystem watcher event handler."""
        self.fs_path = fs_path
        self.timers: dict[Path, threading.Timer] = {}

    def _decode_path(self, path: str | bytes) -> Path:
        """Decode the path."""
        if isinstance(path, bytes):
            return Path(path.decode("utf-8"))
        else:
            return Path(path)

    def on_any_event(self, event: FileSystemEvent):
        """Handle a modified event."""
        if event.is_directory or event.event_type not in ["created", "deleted", "modified", "moved"]:
            return
        src_path = self._decode_path(event.src_path)
        if src_path.is_relative_to(self.fs_path):
            if src_path in self.timers:
                self.timers[src_path].cancel()

            def _publish():
                publish(
                    FileChanged(
                        path=src_path,
                        event_type=ChangeType.DELETE if event.event_type == "deleted" else ChangeType.UPSERT,
                    )
                )
                del self.timers[src_path]

            self.timers[src_path] = threading.Timer(0.5, _publish)  # 0.5s grace period
            self.timers[src_path].start()


class _FSWatchdog:
    def __init__(self, fs_paths: list[Path]):
        """Initialize the filesystem watcher."""
        self.observer = Observer()
        for fs_path in fs_paths:
            self.observer.schedule(_FSWatchdogEventHandler(fs_path), fs_path.as_posix(), recursive=True)

    def start(self):
        """Start the filesystem watcher."""
        self.observer.start()

    def stop(self):
        """Stop the filesystem watcher."""
        self.observer.stop()
        self.observer.join()


class FSWatcher(FrozenBaseModel):
    """Basic filesystem watcher."""

    fs_path: Path | list[Path] = Field(description="Path to watch for changes.")
    supported_types: set[str] | None = Field(default=None, description="Supported file types.")

    def start(self):
        """Setup the filesystem watcher."""
        fs_paths = self.fs_path if isinstance(self.fs_path, list) else [self.fs_path]
        self._fs_watcher = _FSWatchdog(fs_paths)
        self._fs_watcher.start()
        self._running = True
        threading.Thread(target=self._walk_and_publish, daemon=True).start()

    def stop(self):
        """Teardown the filesystem watcher."""
        self._fs_watcher.stop()
        self._running = False

    @field_validator("fs_path")
    @classmethod
    def validate_fs_path(cls, fs_path: Path | list[Path]) -> Path | list[Path]:
        """Validate the filesystem paths."""
        fs_paths = fs_path if isinstance(fs_path, list) else [fs_path]
        fs_paths = list(set(fs_paths))

        for fs_path in fs_paths:
            if not fs_path.is_dir():
                raise ValueError(f"Path {fs_path} is not a directory.")

        # Remove paths that are children of other paths in the list
        filtered_paths = [p for p in fs_paths if not any(p != other and p.is_relative_to(other) for other in fs_paths)]

        return filtered_paths

    def walk_all(self) -> Iterable[Path]:
        """Return all invalidated paths."""
        fs_paths = self.fs_path if isinstance(self.fs_path, list) else [self.fs_path]
        scanned_files = flatten([(p for p in fs_dir.rglob("*") if p.is_file()) for fs_dir in fs_paths])
        scanned_files = (p for p in scanned_files if os.access(p, os.R_OK))

        if self.supported_types is not None:
            scanned_files = (p for p in scanned_files if p.suffix.strip(".") in self.supported_types)
        return scanned_files

    def _walk_and_publish(self):
        for p in self.walk_all():
            if not self._running:
                break
            publish(FileChanged(path=p, event_type=ChangeType.UPSERT))
