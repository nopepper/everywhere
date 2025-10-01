"""Test suite for document indexing functionality."""

from pathlib import Path

import pytest

from everywhere.index.document_index import DocumentIndex, IndexedDocument, walk_all_files


class TestDocumentIndex:
    """Test DocumentIndex for tracking indexed files."""

    @pytest.fixture
    def doc_index(self, temp_workspace: Path) -> DocumentIndex:
        """Doc index fixture."""
        return DocumentIndex(db_path=temp_workspace / "index.db")

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

    def test_add_and_query_document(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Test adding and querying documents."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("test content")
        stat = test_file.stat()

        # Add entry for a provider
        doc_index.add(test_file, stat.st_mtime, stat.st_size, "provider_a")

        # Query rows for this path
        rows = doc_index.get_rows_for_path(test_file)
        assert len(rows) == 1
        assert rows[0] == (stat.st_mtime, stat.st_size, "provider_a")

    def test_has_entry_returns_correct_result(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Test has_entry method."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("test content")
        stat = test_file.stat()

        # Not present initially
        assert not doc_index.has_entry(test_file, stat.st_mtime, stat.st_size, "provider_a")

        # Add it
        doc_index.add(test_file, stat.st_mtime, stat.st_size, "provider_a")

        # Now it's present
        assert doc_index.has_entry(test_file, stat.st_mtime, stat.st_size, "provider_a")

    def test_multiple_providers_same_file(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Test multiple providers can index the same file."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("test content")
        stat = test_file.stat()

        doc_index.add(test_file, stat.st_mtime, stat.st_size, "provider_a")
        doc_index.add(test_file, stat.st_mtime, stat.st_size, "provider_b")

        rows = doc_index.get_rows_for_path(test_file)
        assert len(rows) == 2
        providers = {row[2] for row in rows}
        assert providers == {"provider_a", "provider_b"}

    def test_walk_all_files_function(self, sample_files: tuple[Path, Path, Path]):
        """Test walk_all_files utility function."""
        file1, file2, file3 = sample_files
        directory = file1.parent

        files = list(walk_all_files([directory]))
        assert len(files) == 3
        assert set(files) == {file1, file2, file3}

    def test_walk_all_files_respects_filter(self, temp_workspace: Path):
        """Test that walk_all_files respects the filter."""
        directory = temp_workspace / "filtered"
        directory.mkdir()

        txt_file = directory / "included.txt"
        txt_file.write_text("included")
        md_file = directory / "also_included.md"
        md_file.write_text("also included")
        exe_file = directory / "excluded.exe"
        exe_file.write_text("excluded")

        def path_filter(p: Path) -> bool:
            return p.suffix in {".txt", ".md"}

        files = list(walk_all_files([directory], path_filter))

        assert txt_file in files
        assert md_file in files
        assert exe_file not in files

    def test_get_all_paths(self, doc_index: DocumentIndex, sample_files: tuple[Path, Path, Path]):
        """Test get_all_paths returns unique paths."""
        file1, file2, file3 = sample_files

        for f in sample_files:
            stat = f.stat()
            doc_index.add(f, stat.st_mtime, stat.st_size, "provider_a")
            doc_index.add(f, stat.st_mtime, stat.st_size, "provider_b")

        paths = doc_index.get_all_paths()
        assert len(paths) == 3
        assert paths == {file1, file2, file3}

    def test_persistence_across_instances(self, temp_workspace: Path):
        """Persistence across instances."""
        db_path = temp_workspace / "index.db"
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")
        stat = test_file.stat()

        # First instance: add entry
        doc_index1 = DocumentIndex(db_path=db_path)
        doc_index1.add(test_file, stat.st_mtime, stat.st_size, "provider_x")
        doc_index1.save()
        doc_index1.close()

        # Second instance: should be able to read it
        doc_index2 = DocumentIndex(db_path=db_path)
        rows = doc_index2.get_rows_for_path(test_file)

        assert len(rows) == 1
        assert rows[0] == (stat.st_mtime, stat.st_size, "provider_x")
        doc_index2.close()

    def test_remove_document(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Remove document."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")
        stat = test_file.stat()

        doc_index.add(test_file, stat.st_mtime, stat.st_size, "provider_a")
        assert len(doc_index.get_rows_for_path(test_file)) == 1

        doc_index.remove(test_file, "provider_a")

        assert len(doc_index.get_rows_for_path(test_file)) == 0

    def test_get_all_paths_after_removal(self, doc_index: DocumentIndex, temp_workspace: Path):
        """Test that get_all_paths reflects removals."""
        file1 = temp_workspace / "file1.txt"
        file2 = temp_workspace / "file2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        stat1, stat2 = file1.stat(), file2.stat()
        doc_index.add(file1, stat1.st_mtime, stat1.st_size, "provider_a")
        doc_index.add(file2, stat2.st_mtime, stat2.st_size, "provider_a")

        assert len(doc_index.get_all_paths()) == 2

        doc_index.remove(file1, "provider_a")

        paths = doc_index.get_all_paths()
        assert len(paths) == 1
        assert file2 in paths


class TestIndexedDocument:
    """Test IndexedDocument data class."""

    def test_indexed_document_creation(self, temp_workspace: Path):
        """Test creating an IndexedDocument."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original")

        stat = test_file.stat()
        doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)

        assert doc.path == test_file
        assert doc.last_modified == stat.st_mtime
        assert doc.size == stat.st_size

    def test_indexed_document_equality(self, temp_workspace: Path):
        """Test IndexedDocument comparison."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        stat = test_file.stat()
        doc1 = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)
        doc2 = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)

        assert doc1.path == doc2.path
        assert doc1.last_modified == doc2.last_modified
        assert doc1.size == doc2.size
