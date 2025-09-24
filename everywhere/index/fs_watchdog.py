"""Basic filesystem watcher."""

import threading
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

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
                del self.timers[src_path]

            self.timers[src_path] = threading.Timer(0.5, _publish)  # 0.5s grace period
            self.timers[src_path].start()


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
