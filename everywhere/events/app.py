"""App events."""

from . import Event


class UserSearched(Event):
    """User searched event."""

    query: str
