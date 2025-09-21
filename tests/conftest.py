"""Pytest configuration and fixtures for the test suite."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide the path to the test data directory."""
    return Path("data_test")


@pytest.fixture(scope="session")
def sample_text_files(test_data_dir: Path) -> list[Path]:
    """Provide paths to sample text files for testing."""
    txt_files = list(test_data_dir.glob("*.txt"))
    return txt_files[:4]  # Return first 4 text files


@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Create a temporary workspace for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)
