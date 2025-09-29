"""Test suite for document indexing functionality."""

from pathlib import Path

import pytest

from everywhere.index.document_index import DocumentIndex, IndexedDocument


class TestDocumentIndex:
    """Test DocumentIndex for tracking indexed files."""

    @pytest.fixture
    def doc_index(self, temp_workspace: Path) -> DocumentIndex:
        """Doc index fixture."""
        return DocumentIndex(
            state_path=temp_workspace / "index.pkl",
            path_filter=lambda p: p.suffix in {".txt", ".md"},
        )

    @pytest.fixture
    def sample_files(self, temp_workspace: Path) -> tuple[Path, Path, Path]:
        """Sample files fixture."""
        dir1 = temp_workspace / "dir1"
        dir1.mkdir()

        file1 = dir1 / "file1.txt"
        file1.write_text("content1")

        file2 = dir1 / "file2.txt"
        file2.write_text("content2")

        file3 = dir1 / "file3.md"
        file3.write_text("content3")

        return file1, file2, file3

    def test_get_or_create_new_document(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Get or create new document."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("test content")

        doc = doc_index.get_or_create(test_file)

        assert doc.path == test_file
        assert doc.last_modified > 0
        assert doc.size > 0

    def test_get_returns_none_for_nonexistent(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Get returns none for nonexistent."""
        nonexistent = temp_workspace / "nonexistent.txt"
        doc = doc_index.get(nonexistent)
        assert doc is None

    def test_get_returns_cached_document(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Get returns cached document."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("test content")

        doc1 = doc_index.get_or_create(test_file)
        doc2 = doc_index.get(test_file)

        assert doc1 is doc2

    def test_stale_detection_on_modification(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Test that modified files are detected as stale."""
        import time

        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        doc = doc_index.get_or_create(test_file)
        assert not doc.is_stale()

        time.sleep(0.01)
        test_file.write_text("modified content with different size")

        assert doc.is_stale()

    def test_compute_diff_identifies_new_files(self, doc_index: DocumentIndex, sample_files: tuple[Path, Path, Path]):
        """Compute diff identifies new files."""
        file1, file2, file3 = sample_files
        directory = file1.parent

        upserted, removed = doc_index.compute_diff([directory])

        assert len(upserted) == 3
        assert len(removed) == 0

        paths = {doc.path for doc in upserted}
        assert file1 in paths
        assert file2 in paths
        assert file3 in paths

    def test_compute_diff_identifies_removed_files(
        self, doc_index: DocumentIndex, sample_files: tuple[Path, Path, Path]
    ):
        """Compute diff identifies removed files."""
        file1, file2, file3 = sample_files
        directory = file1.parent

        upserted, removed = doc_index.compute_diff([directory])
        assert len(upserted) == 3

        file2.unlink()

        upserted, removed = doc_index.compute_diff([directory])

        assert len(removed) == 1
        assert removed[0].path == file2

    def test_compute_diff_respects_filter(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Compute diff respects filter."""
        directory = temp_workspace / "filtered"
        directory.mkdir()

        txt_file = directory / "included.txt"
        txt_file.write_text("included")

        exe_file = directory / "excluded.exe"
        exe_file.write_text("excluded")

        upserted, removed = doc_index.compute_diff([directory])

        paths = {doc.path for doc in upserted}
        assert txt_file in paths
        assert exe_file not in paths

    def test_register_and_track_providers(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Register and track providers."""
        doc_index.register_provider("provider_a")
        doc_index.register_provider("provider_b")

        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        doc = doc_index.get_or_create(test_file)

        assert not doc.is_fully_indexed({"provider_a", "provider_b"})

        doc.mark_indexed_by("provider_a")
        assert not doc.is_fully_indexed({"provider_a", "provider_b"})

        doc.mark_indexed_by("provider_b")
        assert doc.is_fully_indexed({"provider_a", "provider_b"})

    def test_persistence_across_instances(self, temp_workspace: Path):
        """Persistence across instances."""
        state_path = temp_workspace / "index.pkl"

        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        doc_index1 = DocumentIndex(state_path=state_path)
        doc1 = doc_index1.get_or_create(test_file)
        doc1.mark_indexed_by("provider_x")
        doc_index1.save()

        doc_index2 = DocumentIndex(state_path=state_path)
        doc2 = doc_index2.get(test_file)

        assert doc2 is not None
        assert doc2.path == test_file
        assert doc2.is_indexed_by("provider_x")

    def test_remove_document(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Remove document."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        doc_index.get_or_create(test_file)
        assert doc_index.get(test_file) is not None

        removed = doc_index.remove(test_file)
        assert removed

        assert doc_index.get(test_file) is None

    def test_indexed_directories_property(self, doc_index: DocumentIndex, sample_files: tuple[Path, Path, Path]):
        """Indexed directories property."""
        file1, file2, file3 = sample_files
        directory = file1.parent

        doc_index.compute_diff([directory])

        indexed_dirs = doc_index.indexed_directories
        assert len(indexed_dirs) > 0


class TestIndexedDocument:
    """Test IndexedDocument data class."""

    def test_is_stale_detects_modification(self, temp_workspace: Path):
        """Is stale detects modification."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        stat = test_file.stat()
        doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)

        assert not doc.is_stale()

        test_file.write_text("modified content")

        assert doc.is_stale()

    def test_is_stale_detects_deletion(self, temp_workspace: Path):
        """Is stale detects deletion."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        stat = test_file.stat()
        doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)

        test_file.unlink()

        assert doc.is_stale()

    def test_provider_tracking(self):
        """Provider tracking."""
        doc = IndexedDocument(path=Path("/fake/path.txt"), last_modified=0.0, size=0)

        assert not doc.is_indexed_by("provider_a")

        doc.mark_indexed_by("provider_a")
        assert doc.is_indexed_by("provider_a")

        doc.mark_indexed_by("provider_b")
        assert doc.is_indexed_by("provider_a")
        assert doc.is_indexed_by("provider_b")

    def test_parsed_text_caching(self):
        """Parsed text caching."""
        doc = IndexedDocument(path=Path("/fake/path.txt"), last_modified=0.0, size=0)

        assert doc.parsed_text is None

        doc.parsed_text = ["chunk1", "chunk2"]
        assert doc.parsed_text == ["chunk1", "chunk2"]
