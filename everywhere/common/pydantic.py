"""Pydantic base model."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class FrozenBaseModel(BaseModel):
    """Pydantic frozen base model."""

    model_config = ConfigDict(frozen=True, strict=True)


class SearchQuery(FrozenBaseModel):
    """Search query."""

    text: str


class SearchResult(FrozenBaseModel):
    """Search result."""

    value: Path
    confidence: float
    size_bytes: int | None = None
    last_modified_ns: int | None = None
