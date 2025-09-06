"""Basic handler interface."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

class ContentHandler(ABC):
    """Handler for content."""