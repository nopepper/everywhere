"""Search provider events."""

from pathlib import Path

from ..common.pydantic import FrozenBaseModel


class FileProcessedEvent(FrozenBaseModel):
    """File processed event."""

    provider_name: str
    success: bool
    path: Path
