"""Test suite for search provider implementations."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from everywhere.common.pydantic import SearchQuery
from everywhere.index.document_index import IndexedDocument
from everywhere.search.tantivy_search import TantivySearchProvider
from everywhere.search.text_embedding_search import EmbeddingSearchProvider


class TestTantivySearchProvider:
    """Test Tantivy full-text search provider."""

    @pytest.fixture
    def provider(self, temp_workspace: Path) -> Generator[TantivySearchProvider, None, None]:
        """Create a Tantivy search provider for testing."""
        provider = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        with provider:
            yield provider

    @pytest.fixture
    def sample_docs(self, temp_workspace: Path) -> list[Path]:
        """Sample docs fixture."""
        docs_dir = temp_workspace / "docs"
        docs_dir.mkdir()

        files = [
            (docs_dir / "python.txt", "Python is a high-level programming language"),
            (docs_dir / "rust.txt", "Rust is a systems programming language"),
            (docs_dir / "java.txt", "Java is an object-oriented programming language"),
            (docs_dir / "ml.txt", "Machine learning with Python and TensorFlow"),
        ]

        for path, content in files:
            path.write_text(content)

        return [path for path, _ in files]

    def test_index_and_search_single_document(self, provider: TantivySearchProvider, temp_workspace: Path):
        """Test indexing and searching a single document."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("The quick brown fox jumps over the lazy dog")

        stat = test_file.stat()
        doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)

        success = provider.index_document(doc)
        assert success

        results = provider.search(SearchQuery(text="fox"))
        assert len(results) == 1
        assert results[0].value == test_file

    def test_search_with_multiple_matching_documents(self, provider: TantivySearchProvider, sample_docs: list[Path]):
        """Test searching returns all matching documents."""
        for doc_path in sample_docs:
            stat = doc_path.stat()
            doc = IndexedDocument(path=doc_path, last_modified=stat.st_mtime, size=stat.st_size)
            provider.index_document(doc)

        results = provider.search(SearchQuery(text="programming"))
        assert len(results) == 3

        result_names = {r.value.name for r in results}
        assert "python.txt" in result_names
        assert "rust.txt" in result_names
        assert "java.txt" in result_names

    def test_search_returns_no_results_for_nonexistent_term(
        self, provider: TantivySearchProvider, sample_docs: list[Path]
    ):
        """Test searching for nonexistent terms returns no results."""
        for doc_path in sample_docs:
            stat = doc_path.stat()
            doc = IndexedDocument(path=doc_path, last_modified=stat.st_mtime, size=stat.st_size)
            provider.index_document(doc)

        results = provider.search(SearchQuery(text="nonexistent"))
        assert len(results) == 0

    def test_empty_query_returns_no_results(self, provider: TantivySearchProvider):
        """Test that empty queries return no results."""
        results = provider.search(SearchQuery(text=""))
        assert len(results) == 0

    def test_document_removal(self, provider: TantivySearchProvider, sample_docs: list[Path]):
        """Test that documents can be removed from the index."""
        for doc_path in sample_docs:
            stat = doc_path.stat()
            doc = IndexedDocument(path=doc_path, last_modified=stat.st_mtime, size=stat.st_size)
            provider.index_document(doc)

        results_before = provider.search(SearchQuery(text="programming"))
        assert len(results_before) == 3

        removed = provider.remove_document(sample_docs[0])
        assert removed

        results_after = provider.search(SearchQuery(text="programming"))
        assert len(results_after) == 2
        assert sample_docs[0] not in [r.value for r in results_after]

    def test_remove_nonexistent_document(self, provider: TantivySearchProvider, temp_workspace: Path):
        """Test removing a document that doesn't exist."""
        nonexistent = temp_workspace / "nonexistent.txt"
        removed = provider.remove_document(nonexistent)
        assert removed

    def test_phrase_query(self, provider: TantivySearchProvider, temp_workspace: Path):
        """Test phrase queries with quotes."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Machine learning is a subset of artificial intelligence")

        stat = test_file.stat()
        doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)
        provider.index_document(doc)

        results = provider.search(SearchQuery(text='"machine learning"'))
        assert len(results) >= 1
        assert results[0].value == test_file

    def test_score_ordering(self, provider: TantivySearchProvider, temp_workspace: Path):
        """Test that documents are ordered by relevance score."""
        file1 = temp_workspace / "highly_relevant.txt"
        file1.write_text("Python Python Python programming")

        file2 = temp_workspace / "less_relevant.txt"
        file2.write_text("Programming in various languages including Python")

        for f in [file1, file2]:
            stat = f.stat()
            doc = IndexedDocument(path=f, last_modified=stat.st_mtime, size=stat.st_size)
            provider.index_document(doc)

        results = provider.search(SearchQuery(text="Python"))
        assert len(results) == 2
        assert results[0].confidence >= results[1].confidence

    def test_index_nonexistent_file_tantivy(self, provider: TantivySearchProvider, temp_workspace: Path):
        """Test indexing a file that doesn't exist fails gracefully."""
        nonexistent = temp_workspace / "nonexistent.txt"

        doc = IndexedDocument(path=nonexistent, last_modified=0.0, size=0)

        success = provider.index_document(doc)
        assert not success

    def test_whitespace_only_query(self, provider: TantivySearchProvider):
        """Test that whitespace-only queries return no results."""
        results = provider.search(SearchQuery(text="   \t\n  "))
        assert len(results) == 0

    def test_reindex_same_document(self, provider: TantivySearchProvider, temp_workspace: Path):
        """Test reindexing the same document updates the index."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("original content")

        stat = test_file.stat()
        doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)
        provider.index_document(doc)

        results = provider.search(SearchQuery(text="original"))
        assert len(results) == 1

        test_file.write_text("modified content")
        stat = test_file.stat()
        doc = IndexedDocument(path=test_file, last_modified=stat.st_mtime, size=stat.st_size)
        provider.index_document(doc)

        results = provider.search(SearchQuery(text="modified"))
        assert len(results) >= 1


class TestEmbeddingSearchProvider:
    """Test embedding-based semantic search provider."""

    @pytest.fixture
    def provider(self, temp_workspace: Path) -> Generator[EmbeddingSearchProvider, None, None]:
        """Create an embedding search provider for testing."""
        provider = EmbeddingSearchProvider(ann_cache_dir=temp_workspace / "ann")
        with provider:
            yield provider

    @pytest.fixture
    def sample_docs(self, temp_workspace: Path) -> list[Path]:
        """Sample docs fixture."""
        docs_dir = temp_workspace / "docs"
        docs_dir.mkdir()

        files = [
            (docs_dir / "dogs.txt", "Dogs are loyal pets that love playing fetch"),
            (docs_dir / "cats.txt", "Cats are independent animals that enjoy napping"),
            (docs_dir / "programming.txt", "Programming requires logical thinking and problem solving"),
        ]

        for path, content in files:
            path.write_text(content)

        return [path for path, _ in files]

    def test_semantic_search_finds_related_concepts(self, provider: EmbeddingSearchProvider, sample_docs: list[Path]):
        """Test that semantic search finds conceptually related documents."""
        for doc_path in sample_docs:
            stat = doc_path.stat()
            doc = IndexedDocument(path=doc_path, last_modified=stat.st_mtime, size=stat.st_size)
            provider.index_document(doc)

        results = provider.search(SearchQuery(text="pets and animals"))
        assert len(results) >= 2

        top_results = {r.value.name for r in results[:2]}
        assert "dogs.txt" in top_results or "cats.txt" in top_results

    def test_document_removal_from_ann_index(self, provider: EmbeddingSearchProvider, sample_docs: list[Path]):
        """Test that documents can be removed from the ANN index."""
        for doc_path in sample_docs:
            stat = doc_path.stat()
            doc = IndexedDocument(path=doc_path, last_modified=stat.st_mtime, size=stat.st_size)
            provider.index_document(doc)

        results_before = provider.search(SearchQuery(text="pets"))
        paths_before = {r.value for r in results_before}

        removed = provider.remove_document(sample_docs[0])
        assert removed

        results_after = provider.search(SearchQuery(text="pets"))
        paths_after = {r.value for r in results_after}

        assert sample_docs[0] in paths_before
        assert sample_docs[0] not in paths_after

    def test_empty_file_handling(self, provider: EmbeddingSearchProvider, temp_workspace: Path):
        """Test that empty files are not indexed."""
        empty_file = temp_workspace / "empty.txt"
        empty_file.write_text("")

        stat = empty_file.stat()
        doc = IndexedDocument(path=empty_file, last_modified=stat.st_mtime, size=stat.st_size)

        success = provider.index_document(doc)
        assert not success

    def test_confidence_scores_normalized(self, provider: EmbeddingSearchProvider, sample_docs: list[Path]):
        """Test that confidence scores are within valid range."""
        for doc_path in sample_docs:
            stat = doc_path.stat()
            doc = IndexedDocument(path=doc_path, last_modified=stat.st_mtime, size=stat.st_size)
            provider.index_document(doc)

        results = provider.search(SearchQuery(text="animals"))

        for result in results:
            assert 0.0 <= result.confidence <= 1.0

    def test_index_nonexistent_file(self, provider: EmbeddingSearchProvider, temp_workspace: Path):
        """Test indexing a file that doesn't exist fails gracefully."""
        nonexistent = temp_workspace / "nonexistent.txt"

        doc = IndexedDocument(path=nonexistent, last_modified=0.0, size=0)

        success = provider.index_document(doc)
        assert not success

    def test_whitespace_only_content(self, provider: EmbeddingSearchProvider, temp_workspace: Path):
        """Test that files with only whitespace are not indexed."""
        whitespace_file = temp_workspace / "whitespace.txt"
        whitespace_file.write_text("   \n\t   \n   ")

        stat = whitespace_file.stat()
        doc = IndexedDocument(path=whitespace_file, last_modified=stat.st_mtime, size=stat.st_size)

        success = provider.index_document(doc)
        assert not success
