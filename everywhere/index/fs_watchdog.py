"""Basic filesystem watcher."""

from collections import defaultdict
from functools import partial
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from everywhere.common.debounce import DebouncedRunner

from ..events import publish
from ..events.watcher import ChangeType, FileChanged


class _FSWatchdogEventHandler(FileSystemEventHandler):
    def __init__(self, fs_path: Path):
        """Initialize the filesystem watcher event handler."""
        self.fs_path = fs_path
        self.debounced: dict[Path, DebouncedRunner] = defaultdict(lambda: DebouncedRunner(0.5))

    def _decode_path(self, path: str | bytes) -> Path:
        """Decode the path."""
        if isinstance(path, bytes):
            return Path(path.decode("utf-8"))
        else:
            return Path(path)

    def _publish(self, event: FileSystemEvent):
        """Publish the event."""
        src_path = self._decode_path(event.src_path)
        try:
            dest_path = self._decode_path(event.dest_path)
        except Exception:
            dest_path = None
        for p in [src_path, dest_path]:
            if p is None:
                continue
            ev_type = ChangeType.UPSERT if p.exists() else ChangeType.REMOVE
            publish(FileChanged(path=p, event_type=ev_type))

    def on_any_event(self, event: FileSystemEvent):
        """Handle a modified event."""
        if event.is_directory or event.event_type not in ["created", "deleted", "modified", "moved"]:
            return
        src_path = self._decode_path(event.src_path)
        if src_path.is_relative_to(self.fs_path):
            self.debounced[src_path].submit(partial(self._publish, event))


class FSWatchdog:
    """Filesystem watchdog."""

    def __init__(self, fs_paths: list[Path]):
        """Initialize the filesystem watcher."""
        self.observer = Observer()
        for fs_path in fs_paths:
            self.observer.schedule(_FSWatchdogEventHandler(fs_path), fs_path.as_posix(), recursive=False)

    def start(self):
        """Start the filesystem watcher."""
        self.observer.start()

    def stop(self):
        """Stop the filesystem watcher."""
        self.observer.stop()
        self.observer.join()
