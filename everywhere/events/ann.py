"""ANN events."""

from . import Event


class IndexSaveStarted(Event):
    """Index save started event."""

    index_size: int
    path_count: int


class IndexSaveFinished(Event):
    """Index save finished event."""
