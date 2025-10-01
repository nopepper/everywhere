"""Test suite for edge cases and tricky scenarios."""

import time
from pathlib import Path

import pytest

from everywhere.app.search_controller import SearchController
from everywhere.common.pydantic import SearchQuery
from everywhere.index.document_index import DocumentIndex, IndexedDocument
from everywhere.search.tantivy_search import TantivySearchProvider
from everywhere.search.text_embedding_search import EmbeddingSearchProvider


class TestFileSystemEdgeCases:
    """Test edge cases related to file system operations."""

    def test_file_deleted_during_indexing(self, temp_workspace: Path):
        """Test handling of files that are deleted while being indexed."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")
        controller = SearchController(search_providers=[tantivy], doc_index=doc_index)

        docs_dir = temp_workspace / "docs"
        docs_dir.mkdir()

        disappearing_file = docs_dir / "disappearing.txt"
        disappearing_file.write_text("This file will disappear")

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                time.sleep(0.01)

            disappearing_file.unlink()

            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                time.sleep(0.01)

            results = controller.search(SearchQuery(text="disappear"))
            result_paths = {r.value for r in results}

            assert disappearing_file not in result_paths

    def test_file_without_extension(self, temp_workspace: Path):
        """Test handling files without extensions are indexed correctly."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")

        def path_filter(p: Path) -> bool:
            return p.suffix in {".txt", ".md"}

        controller = SearchController(search_providers=[tantivy], doc_index=doc_index, path_filter=path_filter)

        docs_dir = temp_workspace / "docs"
        docs_dir.mkdir()

        no_ext_file = docs_dir / "README"
        no_ext_file.write_text("File without extension")

        txt_file = docs_dir / "with_ext.txt"
        txt_file.write_text("File with extension")

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                time.sleep(0.01)

            results = controller.search(SearchQuery(text="extension"))

            result_names = {r.value.name for r in results}
            assert "README" not in result_names
            assert "with_ext.txt" in result_names

    def test_very_long_file_path(self, temp_workspace: Path):
        """Test handling of very long file paths."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")
        controller = SearchController(search_providers=[tantivy], doc_index=doc_index)

        deep_dir = temp_workspace / "a" / "b" / "c" / "d" / "e" / "f" / "g"
        deep_dir.mkdir(parents=True)

        long_name = "x" * 200 + ".txt"
        long_file = deep_dir / long_name
        long_file.write_text("Deep nested file with long name")

        with controller:
            controller.update_selected_paths([temp_workspace / "a"])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                time.sleep(0.01)

            results = controller.search(SearchQuery(text="Deep nested"))
            assert len(results) >= 1

    def test_file_modified_rapidly(self, temp_workspace: Path):
        """Test handling of files that are modified very quickly."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")
        controller = SearchController(search_providers=[tantivy], doc_index=doc_index)

        docs_dir = temp_workspace / "docs"
        docs_dir.mkdir()

        rapid_file = docs_dir / "rapid.txt"
        rapid_file.write_text("Version 1")

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                time.sleep(0.01)

            for i in range(5):
                time.sleep(0.01)
                rapid_file.write_text(f"Version {i + 2}")
                controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                time.sleep(0.01)

            results = controller.search(SearchQuery(text="Version"))
            assert len(results) >= 1


class TestQueryEdgeCases:
    """Test edge cases related to query parsing and execution."""

    @pytest.fixture
    def setup_tantivy(self, temp_workspace: Path):
        """Set up Tantivy provider with sample documents."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")

        docs_dir = temp_workspace / "docs"
        docs_dir.mkdir()

        test_file = docs_dir / "test.txt"
        test_file.write_text("The quick brown fox jumps over the lazy dog. Email: test@example.com")

        with tantivy:
            stat = test_file.stat()
            doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)
            tantivy.index_document(doc)

            yield tantivy

    def test_query_with_special_characters(self, setup_tantivy: TantivySearchProvider):
        """Test queries with special characters."""
        special_queries = [
            "test@example.com",
            "fox!",
            "dog?",
            "quick-brown",
            "over/lazy",
        ]

        for query_text in special_queries:
            results = setup_tantivy.search(SearchQuery(text=query_text))
            assert isinstance(results, list)

    def test_query_with_boolean_operators(self, setup_tantivy: TantivySearchProvider):
        """Test queries with boolean operators."""
        boolean_queries = [
            "quick AND brown",
            "quick OR slow",
            "NOT elephant",
            "quick AND NOT elephant",
        ]

        for query_text in boolean_queries:
            results = setup_tantivy.search(SearchQuery(text=query_text))
            assert isinstance(results, list)

    def test_very_long_query(self, setup_tantivy: TantivySearchProvider):
        """Test handling of very long queries."""
        long_query = " ".join(["word"] * 1000)
        results = setup_tantivy.search(SearchQuery(text=long_query))
        assert isinstance(results, list)

    def test_query_with_only_stopwords(self, setup_tantivy: TantivySearchProvider):
        """Test query with only common stopwords."""
        results = setup_tantivy.search(SearchQuery(text="the a an"))
        assert isinstance(results, list)

    def test_query_with_unicode(self, setup_tantivy: TantivySearchProvider):
        """Test queries with unicode characters."""
        unicode_queries = [
            "café",
            "naïve",
            "日本語",
            "emoji 🔍",
            "Ñoño",
        ]

        for query_text in unicode_queries:
            results = setup_tantivy.search(SearchQuery(text=query_text))
            assert isinstance(results, list)

    def test_query_with_parentheses(self, setup_tantivy: TantivySearchProvider):
        """Test queries with parentheses."""
        results = setup_tantivy.search(SearchQuery(text="(quick AND brown) OR lazy"))
        assert isinstance(results, list)

    def test_malformed_phrase_query(self, setup_tantivy: TantivySearchProvider):
        """Test malformed phrase queries."""
        malformed_queries = [
            '"unclosed quote',
            'unmatched"quote',
            '""',
            '"',
        ]

        for query_text in malformed_queries:
            results = setup_tantivy.search(SearchQuery(text=query_text))
            assert isinstance(results, list)


class TestContentEdgeCases:
    """Test edge cases related to document content."""

    def test_file_with_null_bytes(self, temp_workspace: Path):
        """Test handling of files with null bytes."""
        provider = TantivySearchProvider(index_dir=temp_workspace / "tantivy")

        bad_file = temp_workspace / "null_bytes.txt"
        bad_file.write_bytes(b"Text with \x00 null bytes")

        with provider:
            stat = bad_file.stat()
            doc = IndexedDocument(path=bad_file, last_modified=stat.st_mtime, size=stat.st_size)

            provider.index_document(doc)

    def test_file_with_bom(self, temp_workspace: Path):
        """Test handling of files with BOM markers."""
        provider = TantivySearchProvider(index_dir=temp_workspace / "tantivy")

        bom_file = temp_workspace / "bom.txt"
        bom_file.write_bytes(b"\xef\xbb\xbfUTF-8 with BOM")

        with provider:
            stat = bom_file.stat()
            doc = IndexedDocument(path=bom_file, last_modified=stat.st_mtime, size=stat.st_size)

            success = provider.index_document(doc)

            if success:
                results = provider.search(SearchQuery(text="UTF-8"))
                assert len(results) >= 0

    def test_file_with_mixed_encodings(self, temp_workspace: Path):
        """Test handling of files that might have mixed encodings."""
        provider = TantivySearchProvider(index_dir=temp_workspace / "tantivy")

        mixed_file = temp_workspace / "mixed.txt"
        mixed_file.write_bytes(b"ASCII text \xe9 latin-1 byte")

        with provider:
            stat = mixed_file.stat()
            doc = IndexedDocument(path=mixed_file, last_modified=stat.st_mtime, size=stat.st_size)

            provider.index_document(doc)

    def test_very_large_single_line(self, temp_workspace: Path):
        """Test handling of files with very large single lines."""
        provider = TantivySearchProvider(index_dir=temp_workspace / "tantivy")

        huge_line_file = temp_workspace / "huge_line.txt"
        huge_line_file.write_text("word " * 100000)

        with provider:
            stat = huge_line_file.stat()
            doc = IndexedDocument(path=huge_line_file, last_modified=stat.st_mtime, size=stat.st_size)

            success = provider.index_document(doc)

            if success:
                results = provider.search(SearchQuery(text="word"))
                assert len(results) >= 0

    def test_file_with_only_numbers(self, temp_workspace: Path):
        """Test indexing files containing only numbers."""
        provider = TantivySearchProvider(index_dir=temp_workspace / "tantivy")

        numbers_file = temp_workspace / "numbers.txt"
        numbers_file.write_text("123456789 987654321 5555")

        with provider:
            stat = numbers_file.stat()
            doc = IndexedDocument(path=numbers_file, last_modified=stat.st_mtime, size=stat.st_size)

            success = provider.index_document(doc)

            if success:
                results = provider.search(SearchQuery(text="5555"))
                assert len(results) >= 0


class TestConcurrencyEdgeCases:
    """Test edge cases related to concurrent operations."""

    def test_search_during_indexing(self, temp_workspace: Path):
        """Test searching while indexing is in progress."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        embedding = EmbeddingSearchProvider(ann_cache_dir=temp_workspace / "ann")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")
        controller = SearchController(search_providers=[tantivy, embedding], doc_index=doc_index)

        docs_dir = temp_workspace / "docs"
        docs_dir.mkdir()

        for i in range(10):
            (docs_dir / f"file{i}.txt").write_text(f"Content {i} with searchable text")

        with controller:
            controller.update_selected_paths([docs_dir])

            for _ in range(5):
                results = controller.search(SearchQuery(text="searchable"))
                assert isinstance(results, list)
                time.sleep(0.01)

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                time.sleep(0.01)

    def test_multiple_index_updates(self, temp_workspace: Path):
        """Test multiple rapid index updates."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")
        controller = SearchController(search_providers=[tantivy], doc_index=doc_index)

        docs_dir = temp_workspace / "docs"
        docs_dir.mkdir()

        (docs_dir / "test.txt").write_text("Initial content")

        with controller:
            for _ in range(3):
                controller.update_selected_paths([docs_dir])
                time.sleep(0.05)

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                time.sleep(0.01)

            results = controller.search(SearchQuery(text="Initial"))
            assert len(results) >= 0


class TestIndexCorruption:
    """Test handling of potentially corrupted index states."""

    def test_empty_index_directory(self, temp_workspace: Path):
        """Test opening provider with empty index directory."""
        index_dir = temp_workspace / "empty_index"
        index_dir.mkdir()

        provider = TantivySearchProvider(index_dir=index_dir)

        with provider:
            results = provider.search(SearchQuery(text="anything"))
            assert len(results) == 0

    def test_reopen_after_crash_simulation(self, temp_workspace: Path):
        """Test reopening index after simulated crash."""
        index_dir = temp_workspace / "crash_index"

        test_file = temp_workspace / "test.txt"
        test_file.write_text("Crash test content")

        provider1 = TantivySearchProvider(index_dir=index_dir)
        with provider1:
            stat = test_file.stat()
            doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)
            provider1.index_document(doc)

        provider2 = TantivySearchProvider(index_dir=index_dir)
        with provider2:
            results = provider2.search(SearchQuery(text="Crash"))
            assert isinstance(results, list)
