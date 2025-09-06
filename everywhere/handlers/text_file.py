"""Text file handler."""

from pathlib import Path

from ..base import EmbeddedObject
from ..embedders.text import embed_single

CHUNK_SIZE_CHARS = 5000


def process_text_file(file_path: str) -> list[EmbeddedObject]:
    """Process a text file."""
    content = Path(file_path).read_text()
    return [
        EmbeddedObject(key=embed_single(content[i : i + CHUNK_SIZE_CHARS]), value=file_path)
        for i in range(0, len(content), CHUNK_SIZE_CHARS)
    ]
