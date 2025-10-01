"""Integration tests for end-to-end workflows."""

from pathlib import Path

import pytest

from everywhere.app.search_controller import SearchController
from everywhere.common.pydantic import SearchQuery
from everywhere.index.document_index import DocumentIndex
from everywhere.search.tantivy_search import TantivySearchProvider
from everywhere.search.text_embedding_search import EmbeddingSearchProvider


class TestEndToEndWorkflow:
    """Test complete workflows from indexing to search."""

    @pytest.fixture
    def full_setup(self, temp_workspace: Path) -> tuple[SearchController, Path]:
        """Set up a full search controller with test documents."""
        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        embedding = EmbeddingSearchProvider(ann_cache_dir=temp_workspace / "ann")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")

        def path_filter(p: Path) -> bool:
            return p.suffix in {".txt", ".md", ".py"}

        controller = SearchController(
            search_providers=[tantivy, embedding],
            doc_index=doc_index,
            path_filter=path_filter,
        )

        docs_dir = temp_workspace / "documents"
        docs_dir.mkdir()

        files = {
            "readme.md": "# Project README\nThis project implements machine learning algorithms in Python.",
            "tutorial.md": "## Tutorial\nLearn Python programming step by step with practical examples.",
            "notes.txt": "Python is great for data science and machine learning applications.",
            "script.py": "import numpy as np\nimport pandas as pd\nprint('Hello, World!')",
            "config.json": '{"model": "gpt-4", "temperature": 0.7}',
        }

        for filename, content in files.items():
            (docs_dir / filename).write_text(content)

        return controller, docs_dir

    def test_complete_indexing_and_search_flow(self, full_setup: tuple[SearchController, Path]):
        """Test complete indexing and search workflow."""
        controller, docs_dir = full_setup

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            results = controller.search(SearchQuery(text="Python machine learning"))

            assert len(results) > 0

            result_files = {r.value.name for r in results}
            assert "readme.md" in result_files or "notes.txt" in result_files

    def test_incremental_updates(self, full_setup: tuple[SearchController, Path]):
        """Test incremental updates to the index."""
        controller, docs_dir = full_setup

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            new_file = docs_dir / "new.txt"
            new_file.write_text("Additional Python documentation about neural networks")

            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            results = controller.search(SearchQuery(text="neural networks"))
            result_paths = {r.value for r in results}

            assert new_file in result_paths

    def test_file_deletion_workflow(self, full_setup: tuple[SearchController, Path]):
        """Test file deletion workflow."""
        controller, docs_dir = full_setup

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            target = docs_dir / "notes.txt"
            results_before = controller.search(SearchQuery(text="data science"))
            paths_before = {r.value for r in results_before}

            assert target in paths_before

            target.unlink()

            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            results_after = controller.search(SearchQuery(text="data science"))
            paths_after = {r.value for r in results_after}

            assert target not in paths_after

    def test_dual_search_providers_return_results(self, full_setup: tuple[SearchController, Path]):
        """Test that both search providers return results."""
        controller, docs_dir = full_setup

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            semantic_query = SearchQuery(text="coding tutorials")
            semantic_results = controller.search(semantic_query)

            keyword_query = SearchQuery(text="Python")
            keyword_results = controller.search(keyword_query)

            assert len(semantic_results) > 0
            assert len(keyword_results) > 0

    def test_filtered_file_types(self, full_setup: tuple[SearchController, Path]):
        """Test that filtered file types are not indexed."""
        controller, docs_dir = full_setup

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            results = controller.search(SearchQuery(text="model temperature"))
            result_paths = {r.value for r in results}

            json_file = docs_dir / "config.json"
            assert json_file not in result_paths

    def test_embedding_persistence_and_reload(self, temp_workspace: Path):
        """Test that embedding index persists and reloads correctly."""
        docs_dir = temp_workspace / "documents"
        docs_dir.mkdir()

        test_file = docs_dir / "persistent.txt"
        test_file.write_text("This content should persist across sessions")

        ann_dir = temp_workspace / "ann"
        db_path = temp_workspace / "index.db"

        embedding1 = EmbeddingSearchProvider(ann_cache_dir=ann_dir)
        doc_index1 = DocumentIndex(db_path=db_path)

        controller1 = SearchController(search_providers=[embedding1], doc_index=doc_index1)

        with controller1:
            controller1.update_selected_paths([docs_dir])
            while controller1.indexing_progress[0] != controller1.indexing_progress[1]:
                pass

        embedding2 = EmbeddingSearchProvider(ann_cache_dir=ann_dir)
        doc_index2 = DocumentIndex(db_path=db_path)

        controller2 = SearchController(search_providers=[embedding2], doc_index=doc_index2)

        with controller2:
            results = controller2.search(SearchQuery(text="persist"))
            result_paths = {r.value for r in results}

            assert test_file in result_paths

    def test_multiple_directories(self, temp_workspace: Path):
        """Test indexing multiple directories."""
        dir1 = temp_workspace / "dir1"
        dir2 = temp_workspace / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "file1.txt").write_text("Content in directory one")
        (dir2 / "file2.txt").write_text("Content in directory two")

        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        embedding = EmbeddingSearchProvider(ann_cache_dir=temp_workspace / "ann")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")

        controller = SearchController(search_providers=[tantivy, embedding], doc_index=doc_index)

        with controller:
            controller.update_selected_paths([dir1, dir2])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            results = controller.search(SearchQuery(text="directory"))

            assert len(results) == 2

    def test_special_characters_in_filenames(self, temp_workspace: Path):
        """Test handling files with special characters in names."""
        docs_dir = temp_workspace / "documents"
        docs_dir.mkdir()

        special_files = [
            "file with spaces.txt",
            "file-with-dashes.txt",
            "file_with_underscores.txt",
        ]

        for filename in special_files:
            (docs_dir / filename).write_text(f"Content of {filename}")

        tantivy = TantivySearchProvider(index_dir=temp_workspace / "tantivy")
        embedding = EmbeddingSearchProvider(ann_cache_dir=temp_workspace / "ann")
        doc_index = DocumentIndex(db_path=temp_workspace / "index.db")

        controller = SearchController(search_providers=[tantivy, embedding], doc_index=doc_index)

        with controller:
            controller.update_selected_paths([docs_dir])

            while controller.indexing_progress[0] != controller.indexing_progress[1]:
                pass

            results = controller.search(SearchQuery(text="Content"))

            assert len(results) == len(special_files)
