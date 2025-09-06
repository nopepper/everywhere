"""Pydantic base model."""

from pydantic import BaseModel, ConfigDict


class FrozenBaseModel(BaseModel):
    """Pydantic frozen base model."""

    model_config = ConfigDict(frozen=True, strict=True)
