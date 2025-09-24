"""Text chunking."""

from more_itertools import flatten
from pydantic import BaseModel, Field


class TextChunker(BaseModel):
    """Text chunker."""

    min_chunk_size: int = Field(default=16, description="Minimum chunk size for the text.")
    max_chunk_size: int = Field(default=1024, description="Maximum chunk size for the text.")
    overlap: int = Field(default=128, description="Overlap for the text chunks.")

    def _chunk(self, text: str) -> list[str]:
        """Chunk text."""
        step_size = self.max_chunk_size - self.overlap
        text_chunks = [text[i : i + self.max_chunk_size] for i in range(0, len(text), step_size)]
        text_chunks = [chunk for chunk in text_chunks if len(chunk) >= self.min_chunk_size]
        return text_chunks

    def chunk(self, text: str | list[str]) -> list[str]:
        """Chunk text."""
        if isinstance(text, str):
            return self._chunk(text)
        return list(flatten([self._chunk(t) for t in text]))
