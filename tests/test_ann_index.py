"""Test suite for ANN index functionality."""

from pathlib import Path

import numpy as np
import pytest

from everywhere.index.ann import ANNIndex


class TestANNIndex:
    """Test ANNIndex for approximate nearest neighbor search."""

    @pytest.fixture
    def ann_index(self, temp_workspace: Path) -> ANNIndex:
        """Create an ANN index for testing."""
        return ANNIndex(dims=128, cache_dir=temp_workspace / "ann")

    @pytest.fixture
    def sample_embeddings(self) -> dict[Path, np.ndarray]:
        """Generate sample embeddings for testing."""
        rng = np.random.default_rng(42)
        return {
            Path("/doc1.txt"): rng.random(128, dtype=np.float32),
            Path("/doc2.txt"): rng.random(128, dtype=np.float32),
            Path("/doc3.txt"): rng.random(128, dtype=np.float32),
        }

    def test_add_single_embedding(self, ann_index: ANNIndex, temp_workspace: Path):
        """Test adding a single embedding to the index."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        rng = np.random.default_rng(42)
        embedding = rng.random(128, dtype=np.float32)
        ann_index.add(test_file, embedding)

        assert test_file in ann_index

    def test_add_batch_embeddings(self, ann_index: ANNIndex, temp_workspace: Path):
        """Test adding multiple embeddings at once."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        rng = np.random.default_rng(42)
        embeddings = rng.random((5, 128), dtype=np.float32)
        ann_index.add(test_file, embeddings)

        assert test_file in ann_index

    def test_query_returns_closest_matches(
        self, ann_index: ANNIndex, temp_workspace: Path, sample_embeddings: dict[Path, np.ndarray]
    ):
        """Test querying returns the closest matching documents."""
        for path, emb in sample_embeddings.items():
            temp_file = temp_workspace / path.name
            temp_file.write_text("content")
            ann_index.add(temp_file, emb)

        query_embedding = next(iter(sample_embeddings.values()))
        results = ann_index.query(query_embedding, k=3)

        assert len(results) <= 3
        assert all(isinstance(path, Path) for path, _ in results)
        assert all(isinstance(dist, float) for _, dist in results)

    def test_remove_document(self, ann_index: ANNIndex, temp_workspace: Path):
        """Test removing a document from the index."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        rng = np.random.default_rng(42)
        embedding = rng.random(128, dtype=np.float32)
        ann_index.add(test_file, embedding)

        assert test_file in ann_index

        removed = ann_index.remove(test_file)
        assert removed

        assert test_file not in ann_index

    def test_remove_nonexistent_document(self, ann_index: ANNIndex, temp_workspace: Path):
        """Test removing a document that doesn't exist."""
        nonexistent = temp_workspace / "nonexistent.txt"
        removed = ann_index.remove(nonexistent)
        assert not removed

    def test_clean_removes_invalid_paths(self, ann_index: ANNIndex, temp_workspace: Path):
        """Test that clean removes documents that no longer exist."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        rng = np.random.default_rng(42)
        embedding = rng.random(128, dtype=np.float32)
        ann_index.add(test_file, embedding)

        test_file.unlink()

        removed_count = ann_index.clean()
        assert removed_count >= 1

    def test_persistence(self, temp_workspace: Path):
        """Test that the index persists across instances."""
        cache_dir = temp_workspace / "ann"

        index1 = ANNIndex(dims=128, cache_dir=cache_dir)

        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        rng = np.random.default_rng(42)
        embedding = rng.random(128, dtype=np.float32)
        index1.add(test_file, embedding)
        index1.save()

        index2 = ANNIndex(dims=128, cache_dir=cache_dir)

        assert test_file in index2

    def test_query_empty_index(self, ann_index: ANNIndex):
        """Test querying an empty index returns no results."""
        rng = np.random.default_rng(42)
        query_embedding = rng.random(128, dtype=np.float32)
        results = ann_index.query(query_embedding, k=5)

        assert len(results) == 0

    def test_query_respects_k_limit(self, ann_index: ANNIndex, temp_workspace: Path):
        """Test that query respects the k limit parameter."""
        rng = np.random.default_rng(42)
        for i in range(10):
            test_file = temp_workspace / f"test{i}.txt"
            test_file.write_text(f"content{i}")
            embedding = rng.random(128, dtype=np.float32)
            ann_index.add(test_file, embedding)

        query_embedding = rng.random(128, dtype=np.float32)
        results = ann_index.query(query_embedding, k=5)

        assert len(results) <= 5

    def test_multiple_embeddings_same_document(self, ann_index: ANNIndex, temp_workspace: Path):
        """Test adding multiple embeddings for the same document."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        rng = np.random.default_rng(42)
        embedding1 = rng.random(128, dtype=np.float32)
        ann_index.add(test_file, embedding1)

        embedding2 = rng.random(128, dtype=np.float32)
        ann_index.add(test_file, embedding2)

        assert test_file in ann_index

        query = rng.random(128, dtype=np.float32)
        results = ann_index.query(query, k=10)

        file_results = [path for path, _ in results if path == test_file]
        assert len(file_results) >= 1


class TestANNIndexEdgeCases:
    """Test edge cases and error handling for ANNIndex."""

    def test_add_wrong_shape_embedding(self, temp_workspace: Path):
        """Test adding embedding with wrong shape."""
        ann_index = ANNIndex(dims=128, cache_dir=temp_workspace / "ann")
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        rng = np.random.default_rng(42)
        wrong_embedding = rng.random(256, dtype=np.float32)

        with pytest.raises((ValueError, RuntimeError)):
            ann_index.add(test_file, wrong_embedding)

    def test_mismatched_embedding_dimension(self, temp_workspace: Path):
        """Test that mismatched dimensions raise an error."""
        ann_index = ANNIndex(dims=128, cache_dir=temp_workspace / "ann")

        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")

        rng = np.random.default_rng(42)
        wrong_dim_embedding = rng.random(64, dtype=np.float32)

        with pytest.raises((ValueError, RuntimeError)):
            ann_index.add(test_file, wrong_dim_embedding)

    def test_clean_preserves_valid_paths(self, temp_workspace: Path):
        """Test that clean only removes invalid paths."""
        ann_index = ANNIndex(dims=128, cache_dir=temp_workspace / "ann")

        valid_file = temp_workspace / "valid.txt"
        valid_file.write_text("content")

        invalid_file = temp_workspace / "invalid.txt"
        invalid_file.write_text("content")

        rng = np.random.default_rng(42)
        for f in [valid_file, invalid_file]:
            embedding = rng.random(128, dtype=np.float32)
            ann_index.add(f, embedding)

        invalid_file.unlink()

        ann_index.clean()

        assert valid_file in ann_index
        assert invalid_file not in ann_index
