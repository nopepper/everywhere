"""Basic filesystem watcher."""

from pathlib import Path

import polars as pl
from more_itertools import flatten
from pydantic import Field, field_validator

from ..common.pydantic import FrozenBaseModel
from ..events import publish
from ..events.watcher import ChangeType, FileChanged


class FSWatcher(FrozenBaseModel):
    """Basic filesystem watcher."""

    fs_path: Path | list[Path] = Field(description="Path to watch for changes.")
    db_path: Path | None = Field(
        default=None, description="Path to the database. If not provided, will do everything in memory."
    )

    @field_validator("fs_path")
    @classmethod
    def validate_fs_path(cls, fs_path: Path | list[Path]) -> Path | list[Path]:
        """Validate the filesystem paths."""
        fs_paths = fs_path if isinstance(fs_path, list) else [fs_path]
        fs_paths = list(set(fs_paths))

        for fs_path in fs_paths:
            if not fs_path.is_dir():
                raise ValueError(f"Path {fs_path} is not a directory.")

        # Remove paths that are children of other paths in the list
        filtered_paths = [p for p in fs_paths if not any(p != other and other.is_relative_to(p) for other in fs_paths)]

        return filtered_paths

    def update(self, *, supported_types: list[str] | None = None):
        """Update the database and return all invalidated paths."""
        if self.db_path is None or not self.db_path.exists():
            old_df = pl.DataFrame(
                schema={
                    "path": pl.Utf8,
                    "size": pl.Int64,
                    "mtime_ns": pl.Int64,
                }
            )
        else:
            old_df = pl.read_parquet(self.db_path)

        fs_paths = self.fs_path if isinstance(self.fs_path, list) else [self.fs_path]
        scanned_files = flatten([(p for p in fs_dir.rglob("*") if p.is_file()) for fs_dir in fs_paths])
        if supported_types is not None:
            scanned_files = (p for p in scanned_files if p.suffix.strip(".") in supported_types)
        new_rows = [
            ({"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
            for path in scanned_files
        ]
        new_df = pl.DataFrame(new_rows, schema={"path": pl.Utf8, "size": pl.Int64, "mtime_ns": pl.Int64})

        df_added = new_df.join(old_df, on="path", how="anti")
        df_changed = new_df.join(old_df, on="path", how="inner").filter(
            (pl.col("size") != pl.col("size_right")) | (pl.col("mtime_ns") != pl.col("mtime_ns_right"))
        )
        df_deleted = old_df.join(new_df, on="path", how="anti")

        if self.db_path is not None:
            new_df.write_parquet(self.db_path)

        for path in df_added["path"].to_list():
            publish(FileChanged(path=Path(path), event_type=ChangeType.ADDED))
        for path in df_changed["path"].to_list():
            publish(FileChanged(path=Path(path), event_type=ChangeType.CHANGED))
        for path in df_deleted["path"].to_list():
            publish(FileChanged(path=Path(path), event_type=ChangeType.REMOVED))
