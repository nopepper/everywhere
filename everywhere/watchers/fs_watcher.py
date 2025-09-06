"""Basic filesystem watcher."""

from collections.abc import Iterable
from pathlib import Path
from typing import Self

import polars as pl
from pydantic import Field, model_validator

from ..common.pydantic import EventType, FrozenBaseModel, WatchEvent


class FSWatcher(FrozenBaseModel):
    """Basic filesystem watcher."""

    fs_path: Path = Field(description="Path to watch for changes.")
    db_path: Path | None = Field(
        default=None, description="Path to the database. If not provided, will do everything in memory."
    )

    @model_validator(mode="after")
    def validate_watcher(self) -> Self:
        """Validate the watcher."""
        assert self.fs_path.is_dir()
        return self

    def update(self, *, supported_types: list[str] | None = None) -> Iterable[WatchEvent]:
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

        scanned_files = (p for p in self.fs_path.rglob("*") if p.is_file())
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

        yield from (WatchEvent(value=Path(path), event_type=EventType.ADDED) for path in df_added["path"].to_list())
        yield from (WatchEvent(value=Path(path), event_type=EventType.CHANGED) for path in df_changed["path"].to_list())
        yield from (WatchEvent(value=Path(path), event_type=EventType.REMOVED) for path in df_deleted["path"].to_list())
