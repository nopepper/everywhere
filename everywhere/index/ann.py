"""ANN index helper."""

import json
import math
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from typing import TypedDict

import numpy as np
from voyager import Index, Space, StorageDataType


class IndexedPath(TypedDict):
    """Tracked path."""

    path: str
    last_modified: float
    size: int


class ANNIndex:
    """Index helper."""

    def __init__(self, dims: int, cache_dir: str | Path | None = None):
        """Initialize the index helper, optionally loading from a cache directory."""
        self._dims = dims
        self._index = Index(Space.InnerProduct, num_dimensions=dims, storage_data_type=StorageDataType.E4M3, M=24)
        self._paths_by_id: dict[int, IndexedPath] = {}
        self._ids_by_path: dict[str, list[int]] = defaultdict(list)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.load(self.cache_dir)
            self.clean()

    def __contains__(self, path: Path) -> bool:
        """Check if a path is in the index."""
        if path.as_posix() not in self._ids_by_path:
            return False
        path_id = self._ids_by_path[path.as_posix()][0]
        path_info = self._paths_by_id.get(path_id)
        if path_info is None:
            return False
        new_path_info = {
            "path": path.as_posix(),
            "last_modified": path.stat().st_mtime,
            "size": path.stat().st_size,
        }
        return path_info == new_path_info

    def load(self, source_dir: str | Path) -> None:
        """Load the index and tracked paths from a directory."""
        source_dir = Path(source_dir)
        index_path = source_dir / "index.voyager"
        paths_path = source_dir / "paths.json"

        if not index_path.exists() or not paths_path.exists():
            return

        try:
            self._index = Index.load(str(index_path))
            with open(paths_path) as f:
                paths_data = json.load(f)
                self._paths_by_id = {int(k): v for k, v in paths_data["paths_by_id"].items()}
                self._ids_by_path = defaultdict(list, paths_data["ids_by_path"])
        except Exception:
            # TODO log error
            self._index = Index(
                Space.InnerProduct, num_dimensions=self._dims, storage_data_type=StorageDataType.E4M3, M=24
            )
            self._paths_by_id = {}
            self._ids_by_path = defaultdict(list)

    def save(self, destination_dir: str | Path | None = None) -> None:
        """Save the index and tracked paths atomically.

        If destination_dir is None, saves to the cache_dir set during initialization.
        """
        target_dir = Path(destination_dir) if destination_dir else self.cache_dir
        if not target_dir:
            raise ValueError("No destination directory provided and no cache directory set.")

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target_dir.parent, prefix=f".{target_dir.name}-tmp-") as tmpdir_name:
            tmpdir = Path(tmpdir_name)
            self._index.save(str(tmpdir / "index.voyager"))
            with open(tmpdir / "paths.json", "w") as f:
                json.dump({"paths_by_id": self._paths_by_id, "ids_by_path": self._ids_by_path}, f)

            if target_dir.exists():
                old_dir_temp_name = target_dir.with_name(f"{target_dir.name}-old-{uuid.uuid4().hex}")
                target_dir.rename(old_dir_temp_name)
                tmpdir.rename(target_dir)
                old_dir_temp_name.rename(tmpdir)
            else:
                tmpdir.rename(target_dir)

    def add(self, path: Path, embedding: np.ndarray) -> None:
        """Add an embedding or a batch of embeddings to the index."""
        path_stat = path.stat()
        path_str = path.as_posix()
        path_info: IndexedPath = {
            "path": path_str,
            "last_modified": path_stat.st_mtime,
            "size": path_stat.st_size,
        }

        if len(embedding.shape) == 1:
            new_id = self._index.add_item(embedding)
            self._paths_by_id[new_id] = path_info
            self._ids_by_path[path_str].append(new_id)
        else:
            assert len(embedding.shape) == 2
            new_ids = self._index.add_items(embedding)
            for new_id in new_ids:
                self._paths_by_id[new_id] = path_info
            self._ids_by_path[path_str].extend(new_ids)

    def remove(self, path: Path) -> bool:
        """Remove all embeddings associated with a path from the index."""
        path_str = path.as_posix()
        if path_str not in self._ids_by_path:
            return False

        ids_to_remove = self._ids_by_path.pop(path_str)
        for ann_id in ids_to_remove:
            if ann_id in self._paths_by_id:
                del self._paths_by_id[ann_id]
        return True

    def query(self, embedding: np.ndarray, k: int) -> list[tuple[Path, float]]:
        """Query the index."""
        if self._index.num_elements == 0:
            return []
        k = min(k, self._index.num_elements)
        try:
            ids, distances = self._index.query(embedding, k=k)
        except Exception:
            # In some edge cases, we can't get the exact number of elements in the index
            ids, distances = self._index.query(embedding, k=math.ceil(k * 0.8))
        results: list[tuple[Path, float]] = []
        for path_id, distance in zip(ids, distances, strict=True):
            path_info = self._paths_by_id.get(path_id)
            if path_info:
                results.append((Path(path_info["path"]), float(distance)))
        return results

    def clean(self) -> int:
        """Clean the index and remove invalid or outdated paths."""
        removed = 0
        for path_str in list(self._ids_by_path.keys()):
            path = Path(path_str)
            ids = self._ids_by_path[path_str]
            if not ids:
                self.remove(path)
                removed += 1
                continue

            first_id = ids[0]
            path_info = self._paths_by_id.get(first_id)

            if not path.exists() or path_info is None:
                self.remove(path)
                removed += 1
                continue

            stat = path.stat()
            if stat.st_mtime != path_info["last_modified"] or stat.st_size != path_info["size"]:
                self.remove(path)
                removed += 1
                continue

        # Re-create the index if there are IDs to remove
        current_ids = set(self._paths_by_id.keys())
        loaded_ids = set(self._index.ids)
        ids_to_remove = loaded_ids - current_ids
        if len(ids_to_remove) > 0:
            recreated = Index(
                self._index.space,
                self._index.num_dimensions,
                self._index.M,
                self._index.ef_construction,
                max_elements=len(self._index),
                storage_data_type=self._index.storage_data_type,
            )
            ordered_ids = list(current_ids)
            if ordered_ids:
                recreated.add_items(self._index.get_vectors(ordered_ids), ordered_ids)
            self._index = recreated

        return removed
