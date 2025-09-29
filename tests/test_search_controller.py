"""Test suite for search controller orchestration."""

from collections.abc import Generator
from pathlib import Path

import pytest

from everywhere.app.search_controller import SearchController, normalize_results, normalize_scores
from everywhere.common.pydantic import SearchQuery, SearchResult
from everywhere.index.document_index import DocumentIndex
from everywhere.search.tantivy_search import TantivySearchProvider
from everywhere.search.text_embedding_search import EmbeddingSearchProvider


class TestSearchController:
    """Test SearchController for managing multiple search providers."""

    @pytest.fixture
    def controller(self, temp_workspace: Path) -> Generator[SearchController, None, None]:
        """Controller fixture."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        embedding = EmbeddingSearchProvider(ann_cache_dir=temp_workspace / "ann")

        doc_index = DocumentIndex(
            state_path=temp_workspace / "index.pkl",
            path_filter=lambda p: p.suffix in {".txt", ".md"},
        )

        controller = SearchController(search_providers=[tantivy, embedding], doc_index=doc_index)

        with controller:
            yield controller

    @pytest.fixture
    def sample_directory(self, temp_workspace: Path) -> Path:
        """Sample directory fixture."""
        docs_dir = temp_workspace / "documents"
        docs_dir.mkdir()

        files = [
            (docs_dir / "python.txt", "Python programming language is versatile"),
            (docs_dir / "rust.txt", "Rust systems programming language"),
            (docs_dir / "java.txt", "Java object-oriented language"),
        ]

        for path, content in files:
            path.write_text(content)

        return docs_dir

    def test_initial_indexing(self, controller: SearchController, sample_directory: Path):
        """Initial indexing."""
        controller.update_selected_paths([sample_directory])

        progress = controller.indexing_progress
        assert progress[0] > 0

    def test_search_aggregates_results(self, controller: SearchController, sample_directory: Path):
        """Search aggregates results."""
        controller.update_selected_paths([sample_directory])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        results = controller.search(SearchQuery(text="programming"))

        assert len(results) > 0

        for result in results:
            assert 0.0 <= result.confidence <= 1.0

    def test_directory_removal_triggers_cleanup(self, controller: SearchController, sample_directory: Path):
        """Directory removal triggers cleanup."""
        controller.update_selected_paths([sample_directory])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        results_before = controller.search(SearchQuery(text="Python"))
        assert len(results_before) > 0

        controller.update_selected_paths([])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        results_after = controller.search(SearchQuery(text="Python"))
        assert len(results_after) == 0

    def test_file_removal_reflected_in_search(self, controller: SearchController, sample_directory: Path):
        """File removal reflected in search."""
        controller.update_selected_paths([sample_directory])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        target_file = sample_directory / "python.txt"
        target_file.unlink()

        controller.update_selected_paths([sample_directory])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        results = controller.search(SearchQuery(text="Python"))
        result_paths = {r.value for r in results}

        assert target_file not in result_paths

    def test_indexed_paths_tracking(self, controller: SearchController, sample_directory: Path):
        """Indexed paths tracking."""
        controller.update_selected_paths([sample_directory])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        indexed = controller.indexed_paths
        assert len(indexed) > 0

    def test_empty_query_returns_empty_results(self, controller: SearchController, sample_directory: Path):
        """Empty query returns empty results."""
        controller.update_selected_paths([sample_directory])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        results = controller.search(SearchQuery(text=""))
        assert len(results) == 0

    def test_reindexing_modified_file(self, controller: SearchController, sample_directory: Path):
        """Reindexing modified file."""
        controller.update_selected_paths([sample_directory])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        target_file = sample_directory / "python.txt"
        target_file.write_text("Python machine learning artificial intelligence")

        controller.update_selected_paths([sample_directory])

        while controller.indexing_progress[0] != controller.indexing_progress[1]:
            pass

        results = controller.search(SearchQuery(text="machine learning"))
        result_paths = {r.value for r in results}

        assert target_file in result_paths


class TestNormalizationFunctions:
    """Test score and result normalization functions."""

    def test_normalize_scores_empty_list(self):
        """Normalize scores empty list."""
        result = normalize_scores([])
        assert result == []

    def test_normalize_scores_single_value(self):
        """Normalize scores single value."""
        result = normalize_scores([0.5])
        assert len(result) == 1
        assert 0.0 <= result[0] <= 1.0

    def test_normalize_scores_range(self):
        """Normalize scores range."""
        scores = [0.1, 0.5, 0.9]
        result = normalize_scores(scores)

        assert len(result) == 3
        for score in result:
            assert 0.0 <= score <= 1.0

        assert result[2] >= result[1] >= result[0]

    def test_normalize_results_deduplicates(self):
        """Normalize results deduplicates."""
        path1 = Path("test.txt")
        path2 = Path("other.txt")

        results = [
            SearchResult(value=path1, confidence=0.8),
            SearchResult(value=path1, confidence=0.6),
            SearchResult(value=path2, confidence=0.7),
        ]

        normalized = normalize_results(results)

        paths = [r.value for r in normalized]
        assert len(paths) == len(set(paths))

    def test_normalize_results_keeps_highest_score(self):
        """Normalize results keeps highest score."""
        path1 = Path("test.txt")

        results = [
            SearchResult(value=path1, confidence=0.3),
            SearchResult(value=path1, confidence=0.9),
            SearchResult(value=path1, confidence=0.5),
        ]

        normalized = normalize_results(results)

        assert len(normalized) == 1
        assert normalized[0].value == path1

    def test_normalize_results_sorts_by_confidence(self):
        """Normalize results sorts by confidence."""
        results = [
            SearchResult(value=Path("low.txt"), confidence=0.2),
            SearchResult(value=Path("high.txt"), confidence=0.9),
            SearchResult(value=Path("medium.txt"), confidence=0.5),
        ]

        normalized = normalize_results(results)

        for i in range(len(normalized) - 1):
            assert normalized[i].confidence >= normalized[i + 1].confidence
