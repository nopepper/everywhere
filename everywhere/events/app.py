"""App events."""

from pathlib import Path

from . import Event


class UserSearched(Event):
    """User searched event."""

    query: str


class UserSelectedDirectories(Event):
    """User selected directories event."""

    directories: list[Path]


class AppResized(Event):
    """App resized event."""

    width: int
    height: int
