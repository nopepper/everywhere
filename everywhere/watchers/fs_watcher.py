"""Basic filesystem watcher."""

import os
import threading
from collections.abc import Iterable
from pathlib import Path

from more_itertools import flatten
from pydantic import BaseModel, Field, field_validator
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from ..events import add_callback, publish
from ..events.app import UserSelectedDirectories
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
                src_path = self._decode_path(event.src_path)
                try:
                    dest_path = self._decode_path(event.dest_path)
                except Exception:
                    dest_path = None
                for p in [src_path, dest_path]:
                    if p is None:
                        continue
                    ev_type = ChangeType.UPSERT if p.exists() else ChangeType.DELETE
                    publish(FileChanged(path=p, event_type=ev_type))
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


class FSWatcher(BaseModel):
    """Basic filesystem watcher."""

    fs_path: list[Path] = Field(default=[], description="Path to watch for changes.")
    supported_types: set[str] | None = Field(default=None, description="Supported file types.")

    def start(self):
        """Setup the filesystem watcher."""
        self._fs_watcher = _FSWatchdog(self.fs_path)
        self._fs_watcher.start()
        self._running = True
        threading.Thread(target=self._walk_and_publish, daemon=True).start()
        # Listen for directory selection events
        add_callback(UserSelectedDirectories, self._on_directories_selected)

    def stop(self):
        """Teardown the filesystem watcher."""
        self._fs_watcher.stop()
        self._running = False

    def _on_directories_selected(self, event: UserSelectedDirectories) -> None:
        """Handle user selected directories event."""
        # Convert string paths to Path objects
        new_paths = event.directories

        # Only restart if paths have actually changed
        if set(new_paths) != set(self.fs_path):
            self._restart_with_new_paths(new_paths)

    def _restart_with_new_paths(self, new_paths: list[Path]) -> None:
        """Restart the watcher with new paths."""
        # Get files that are no longer being watched
        removed_files = self._get_removed_files(new_paths)

        # Emit DELETE events for files that are no longer watched
        for file_path in removed_files:
            publish(FileChanged(path=file_path, event_type=ChangeType.DELETE))

        # Stop current watcher
        self._fs_watcher.stop()
        self._running = False

        # Update paths
        self.fs_path = new_paths

        # Start new watcher
        self._fs_watcher = _FSWatchdog(self.fs_path)
        self._fs_watcher.start()
        self._running = True
        threading.Thread(target=self._walk_and_publish, daemon=True).start()

    def _get_removed_files(self, new_paths: list[Path]) -> set[Path]:
        """Get files that were previously watched but are no longer watched."""
        if not self._running:
            return set()

        # Get all files currently being watched
        old_files = set(self.walk_all())

        # Get all files that will be watched with new paths
        temp_watcher = FSWatcher(fs_path=new_paths, supported_types=self.supported_types)
        new_files = set(temp_watcher.walk_all())

        # Return files that were watched before but won't be watched anymore
        return old_files - new_files

    @field_validator("fs_path")
    @classmethod
    def validate_fs_path(cls, fs_path: list[Path]) -> list[Path]:
        """Validate the filesystem paths."""
        fs_path = list(set(fs_path))

        for p in fs_path:
            if not p.is_dir():
                raise ValueError(f"Path {p} is not a directory.")

        # Remove paths that are children of other paths in the list
        filtered_paths = [p for p in fs_path if not any(p != other and p.is_relative_to(other) for other in fs_path)]

        return filtered_paths

    def walk_all(self) -> Iterable[Path]:
        """Return all invalidated paths."""
        scanned_files = flatten([(p for p in fs_dir.rglob("*") if p.is_file()) for fs_dir in self.fs_path])
        scanned_files = (p for p in scanned_files if os.access(p, os.R_OK))

        if self.supported_types is not None:
            scanned_files = (p for p in scanned_files if p.suffix.strip(".") in self.supported_types)
        return scanned_files

    def _walk_and_publish(self):
        for p in self.walk_all():
            if not self._running:
                break
            publish(FileChanged(path=p, event_type=ChangeType.UPSERT))
