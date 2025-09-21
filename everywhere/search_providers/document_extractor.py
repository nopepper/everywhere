"""Document extractor protocol for dependency injection."""

from pathlib import Path
from typing import Protocol

from markitdown import StreamInfo


class DocumentExtractor(Protocol):
    """Protocol for extracting text from documents."""

    def extract(self, path: Path, stream_info: StreamInfo | None = None) -> str:
        """Return extracted text for ``path`` using optional stream metadata."""
        ...
