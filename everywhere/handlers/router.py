"""Router for handling different file types."""

from pathlib import Path

from tqdm.auto import tqdm

from ..base import EmbeddedObject
from .text_file import process_text_file


def process_files(path_str: str) -> list[EmbeddedObject]:
    """Process a file."""
    path = Path(path_str)
    if path.is_file():
        if path.suffix == ".txt":
            return process_text_file(path.as_posix())
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")
    else:
        results = []
        for p in tqdm(list(path.iterdir()), desc="Processing tree"):
            results.extend(process_files(p.as_posix()))
        return results
